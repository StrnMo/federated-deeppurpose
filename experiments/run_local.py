"""
Local-only baseline: each client trains its own model on its own data, with no communication.

Each model is selected on its own client's validation set and evaluated on its own test set
and on the global test set (all clients' drugs). Results go to <output-dir>/client_<k>/ and
<output-dir>/local_summary.csv.

Example:
    python experiments/run_local.py --epochs 100 --output-dir outputs/local --resume
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.engine import add_common_args, get_device, set_seed, prepare_run_dir
from utils.features import featurize_protocol
from utils.trainer import train_supervised


def main(argv=None):
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args(argv)

    prepare_run_dir(args, has_checkpoint=False)
    device = get_device(args.device)
    print(f"Local-only training on {device}, output: {args.output_dir}")

    data = featurize_protocol(num_clients=args.num_clients, seed=args.split_seed, subsample=args.subsample)
    g = data["global"]

    summary = []
    for k, client in enumerate(data["clients"]):
        out_dir = os.path.join(args.output_dir, f"client_{k}")
        ckpt = os.path.join(out_dir, "last.pt")
        if os.path.exists(ckpt) and not args.resume:
            raise SystemExit(f"{ckpt} exists. Pass --resume to continue it, or choose a new --output-dir.")
        ckpt = ckpt if os.path.exists(ckpt) else None

        print(f"\n=== Client {k}: train {len(client['train'])}  val {len(client['val'])}  test {len(client['test'])}")
        set_seed(args.seed)
        result = train_supervised(args, client["train"], g["val"], g["test"], out_dir, device,
                                  select_clients=[k], ckpt_path=ckpt)

        own, glob = result["test"][str(k)], result["test"]["all"]
        summary.append({"client": k, "n_train": len(client["train"]), "best_epoch": result["best_epoch"],
                        **{f"own_test_{m}": own[m] for m in ("mse", "pearson", "ci", "r2")},
                        **{f"global_test_{m}": glob[m] for m in ("mse", "pearson", "ci", "r2")}})

    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(args.output_dir, "local_summary.csv"), index=False)
    print("\n" + summary.round(4).to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
