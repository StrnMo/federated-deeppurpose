"""
Shared evaluation protocol for centralized, local-only and FedAvg experiments.

1. Drugs are partitioned into disjoint client sets (utils.data_split).
2. Each client's drug-target pairs are split at random into train/val/test.
3. Global val/test = union of client val/test sets.
   Centralized training uses the union of client train sets.

The split is stored as indices into load_davis() order so every experiment
uses exactly the same pairs.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.load_davis import load_davis
from utils.data_split import assign_drugs_to_clients

SPLIT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "splits")
SPLITS = ("train", "val", "test")


def to_pkd(y):
    """Convert Kd in nM to pKd = -log10(Kd [M])."""
    y = np.maximum(np.asarray(y, dtype=np.float64), 1e-10)
    return -np.log10(y * 1e-9)


def _paths(seed):
    base = os.path.join(SPLIT_DIR, f"protocol_seed{seed}")
    return base + ".csv", base + ".json"


def make_protocol(num_clients=5, frac=(0.7, 0.1, 0.2), seed=42, save=True):
    """Build the client/split assignment for every pair. Returns a DataFrame (idx, client, split)."""
    assert abs(sum(frac) - 1.0) < 1e-9, "frac must sum to 1"

    drugs, targets, y = load_davis()
    client_drug_sets = assign_drugs_to_clients(drugs, num_clients, seed)

    rows = []
    for c, client_drugs in enumerate(client_drug_sets):
        idx = np.where(np.isin(drugs, list(client_drugs)))[0]
        idx = np.random.RandomState(seed + c).permutation(idx)

        n_train = int(round(frac[0] * len(idx)))
        n_val = int(round(frac[1] * len(idx)))
        parts = {
            "train": idx[:n_train],
            "val": idx[n_train:n_train + n_val],
            "test": idx[n_train + n_val:],
        }
        for split, part in parts.items():
            rows.append(pd.DataFrame({"idx": np.sort(part), "client": c, "split": split}))

    assignment = pd.concat(rows, ignore_index=True)
    _check(assignment, drugs, len(y))

    if save:
        csv_path, json_path = _paths(seed)
        os.makedirs(SPLIT_DIR, exist_ok=True)
        assignment.to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            json.dump(_summary(assignment, y, num_clients, frac, seed), f, indent=2)

    return assignment


def _check(assignment, drugs, n_total):
    """Sanity checks: every pair used exactly once, no drug shared across clients."""
    assert assignment["idx"].is_unique, "a pair appears in more than one split"
    assert len(assignment) == n_total, "not every pair is assigned"
    drug_owner = pd.Series(drugs[assignment["idx"].values]).groupby(assignment["client"].values).unique()
    seen = set()
    for client_drugs in drug_owner:
        assert seen.isdisjoint(client_drugs), "a drug is shared between clients"
        seen.update(client_drugs)


def _summary(assignment, y, num_clients, frac, seed):
    pkd = to_pkd(y)
    stats = {}
    for (c, split), g in assignment.groupby(["client", "split"]):
        v = pkd[g["idx"].values]
        stats.setdefault(f"client_{c}", {})[split] = {
            "n": int(len(v)),
            "pkd_mean": round(float(v.mean()), 4),
            "pkd_std": round(float(v.std()), 4),
            "frac_pkd_5": round(float(np.mean(np.isclose(v, 5.0))), 4),
        }
    return {"num_clients": num_clients, "frac": list(frac), "seed": seed, "stats": stats}


def load_protocol(num_clients=5, frac=(0.7, 0.1, 0.2), seed=42):
    """
    Load (or create) the protocol and return DataFrames with columns
    idx, Drug, Target, Label (pKd), client.

    Returns:
        {
            'clients': [ {'train': df, 'val': df, 'test': df}, ... ],
            'global':  {'train': df, 'val': df, 'test': df},
        }
    """
    csv_path, json_path = _paths(seed)
    drugs, targets, y = load_davis()

    if os.path.exists(csv_path) and os.path.exists(json_path):
        with open(json_path) as f:
            meta = json.load(f)
        if meta["num_clients"] != num_clients or not np.allclose(meta["frac"], frac):
            raise ValueError(f"{csv_path} was built with num_clients={meta['num_clients']}, "
                             f"frac={meta['frac']}; delete it or pass matching arguments")
        assignment = pd.read_csv(csv_path)
        _check(assignment, drugs, len(y))
    else:
        assignment = make_protocol(num_clients, frac, seed)

    pkd = to_pkd(y)
    full = pd.DataFrame({"idx": assignment["idx"].values,
                         "Drug": drugs[assignment["idx"].values],
                         "Target": targets[assignment["idx"].values],
                         "Label": pkd[assignment["idx"].values],
                         "client": assignment["client"].values,
                         "split": assignment["split"].values})

    clients = []
    for c in range(num_clients):
        clients.append({s: full[(full["client"] == c) & (full["split"] == s)]
                        .drop(columns="split").reset_index(drop=True) for s in SPLITS})
    global_sets = {s: full[full["split"] == s].drop(columns="split").reset_index(drop=True)
                   for s in SPLITS}

    return {"clients": clients, "global": global_sets}


def subsample_protocol(proto, frac, seed=42):
    """Keep a random fraction of every client split (for quick tests); rebuild global sets."""
    clients = [{s: d[s].sample(frac=frac, random_state=seed).reset_index(drop=True) for s in SPLITS}
               for d in proto["clients"]]
    global_sets = {s: pd.concat([d[s] for d in clients], ignore_index=True) for s in SPLITS}
    return {"clients": clients, "global": global_sets}


if __name__ == "__main__":
    print("=" * 60)
    print("Building evaluation protocol")
    print("=" * 60)
    make_protocol()
    proto = load_protocol()
    csv_path, json_path = _paths(42)

    print(f"\nSaved: {csv_path}\n       {json_path}\n")
    header = f"{'set':<10}{'drugs':>6}{'train':>8}{'val':>7}{'test':>7}{'test pKd mean±std':>21}{'test %pKd=5':>13}"
    print(header)
    print("-" * len(header))
    named = [(f"client {c}", d) for c, d in enumerate(proto["clients"])] + [("global", proto["global"])]
    for name, d in named:
        t = d["test"]["Label"]
        n_drugs = pd.concat([d[s] for s in SPLITS])["Drug"].nunique()
        print(f"{name:<10}{n_drugs:>6}{len(d['train']):>8}{len(d['val']):>7}{len(d['test']):>7}"
              f"{t.mean():>13.3f} ± {t.std():.3f}{100 * np.isclose(t, 5.0).mean():>12.1f}%")
    print("\nChecks passed: pairs disjoint across splits, all pairs used, drugs disjoint across clients.")
