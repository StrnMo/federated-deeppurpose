"""
Check that the cached protein encoding (utils.features.DTIDataset) gives exactly the same
batches, predictions and trained weights as DeepPurpose's data_process_loader, and time both.

    python tests/verify_protein_cache.py [--n 2048] [--device cpu]
"""

import os
import sys
import time
import argparse
import torch
from torch.utils import data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DeepPurpose.utils import data_process_loader, mpnn_collate_func, generate_config
from utils.engine import MODEL_CONFIG, get_device, build_model, train_one_epoch, predict
from utils.features import featurize_protocol, DTIDataset, make_loader


def deeppurpose_loader(df, batch_size, shuffle=False, seed=0):
    ds = data_process_loader(df.index.values, df.Label.values, df, **generate_config(**MODEL_CONFIG))
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen, drop_last=False,
                           collate_fn=mpnn_collate_func)


def same_batches(a, b):
    for (d1, p1, y1), (d2, p2, y2) in zip(a, b):
        if not torch.equal(p1.float(), p2.float()) or not torch.equal(y1.float(), y2.float()):
            return False
        if not all(torch.equal(x1, x2) for x1, x2 in zip(d1, d2)):
            return False
    return True


def timed(fn):
    t = time.time()
    out = fn()
    return out, time.time() - t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2048, help="number of pairs to compare")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = get_device(args.device)
    df = featurize_protocol()["global"]["train"].iloc[:args.n].reset_index(drop=True)
    bs = args.batch_size
    print(f"Comparing on {len(df)} pairs, device {device}\n")

    # 1. identical batches
    ok_batches = same_batches(deeppurpose_loader(df, bs), make_loader(DTIDataset(df), bs))
    print(f"1. batches identical:            {ok_batches}")

    # 2. identical predictions from the same initial model
    model = build_model(device, seed=0)
    p_dp, t_pred_dp = timed(lambda: predict(model, deeppurpose_loader(df, bs), device))
    p_ca, t_pred_ca = timed(lambda: predict(model, make_loader(DTIDataset(df), bs), device))
    ok_pred = bool((p_dp == p_ca).all())
    print(f"2. predictions identical:        {ok_pred}  (max |diff| = {abs(p_dp - p_ca).max():.3g})")

    # 3. identical weights after one training epoch with the same shuffle and dropout seeds
    def train(loader):
        m = build_model(device, seed=0)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        torch.manual_seed(123)
        train_one_epoch(m, loader, opt, device)
        return m.state_dict()

    w_dp, t_train_dp = timed(lambda: train(deeppurpose_loader(df, bs, shuffle=True, seed=7)))
    w_ca, t_train_ca = timed(lambda: train(make_loader(DTIDataset(df), bs, shuffle=True, seed=7)))
    ok_train = all(torch.equal(w_dp[k], w_ca[k]) for k in w_dp)
    print(f"3. trained weights identical:    {ok_train}")

    # 4. data loading alone
    _, t_load_dp = timed(lambda: sum(1 for _ in deeppurpose_loader(df, bs)))
    _, t_load_ca = timed(lambda: sum(1 for _ in make_loader(DTIDataset(df), bs)))

    print(f"\nTiming for {len(df)} pairs        DeepPurpose   cached   speed-up")
    for name, a, b in [("data loading only", t_load_dp, t_load_ca), ("prediction", t_pred_dp, t_pred_ca),
                       ("training epoch", t_train_dp, t_train_ca)]:
        print(f"  {name:<26}{a:>9.1f}s{b:>8.1f}s{a / b:>9.1f}x")

    if not (ok_batches and ok_pred and ok_train):
        sys.exit("FAILED: cached encoding differs from DeepPurpose")
    print("\nPASSED: the protein cache changes no batches, predictions or weights.")


if __name__ == "__main__":
    main()
