"""
Shared training utilities: device, model, train/evaluate, metrics, checkpoints, CSV logs.
Used by the centralized, local-only and FedAvg experiments.
"""

import os
import csv
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from lifelines.utils import concordance_index

import DeepPurpose.encoders as dp_encoders
from DeepPurpose.utils import generate_config
from DeepPurpose.encoders import MPNN, CNN
from DeepPurpose.DTI import Classifier

# Same architecture as the original baseline scripts
MODEL_CONFIG = dict(
    drug_encoding="MPNN",
    target_encoding="CNN",
    mpnn_hidden_size=128,
    mpnn_depth=3,
    cnn_target_filters=[32, 64, 96],
    cnn_target_kernels=[4, 8, 12],
)

# Arguments that may change when resuming a run
RESUMABLE_ARGS = {"resume", "output_dir", "epochs", "rounds", "device", "num_workers"}


# ---------------------------------------------------------------- setup

def add_common_args(parser):
    parser.add_argument("--output-dir", required=True, help="run directory (e.g. on Google Drive)")
    parser.add_argument("--resume", action="store_true",
                        help="continue from <output-dir>/last.pt if it exists, otherwise start fresh")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42, help="model init / shuffling seed")
    parser.add_argument("--split-seed", type=int, default=42, help="client partition and train/val/test split seed")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--subsample", type=float, default=None,
                        help="keep this fraction of every split (quick tests only)")
    parser.add_argument("--init-bias", default="label_mean", choices=["label_mean", "none"],
                        help="initialise the output-layer bias to the training-label mean")
    return parser


def get_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    device = torch.device(name)
    # DeepPurpose's MPNN moves its inputs to this module-level device
    dp_encoders.device = device
    return device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(device, seed, label_mean=None):
    """
    MPNN-CNN Classifier, built like DeepPurpose's DBTA but without its file side effects.
    If label_mean is given, the output-layer bias starts at it, so the untrained model predicts
    roughly the mean pKd instead of ~0.
    """
    torch.manual_seed(seed)
    config = generate_config(**MODEL_CONFIG)
    model = Classifier(MPNN(config["hidden_dim_drug"], config["mpnn_depth"]),
                       CNN("protein", **config), **config)
    if label_mean is not None:
        with torch.no_grad():
            model.predictor[-1].bias.fill_(float(label_mean))
    return model.to(device)


# ---------------------------------------------------------------- train / evaluate

def train_one_epoch(model, loader, opt, device):
    """One pass over `loader`; returns the sample-weighted mean training MSE."""
    model.train()
    total, n = 0.0, 0
    for v_d, v_p, label in loader:
        v_p = v_p.float().to(device)
        label = label.float().to(device)
        score = model(v_d, v_p)
        loss = F.mse_loss(torch.squeeze(score, 1), label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(label)
        n += len(label)
    return total / max(n, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for v_d, v_p, _ in loader:
        score = model(v_d, v_p.float().to(device))
        preds.append(torch.squeeze(score, 1).cpu())
    model.train()
    return torch.cat(preds).numpy() if preds else np.array([])


def regression_metrics(y, p):
    """MSE, Pearson, concordance index, and R2 = 1 - MSE / MSE(predict the mean)."""
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    mse = float(np.mean((y - p) ** 2))
    var = float(np.var(y))
    try:
        pearson = float(pearsonr(y, p)[0])
    except Exception:
        pearson = float("nan")
    try:
        ci = float(concordance_index(y, p))
    except Exception:
        ci = float("nan")
    return {"n": int(len(y)), "mse": mse, "pearson": pearson, "ci": ci,
            "r2": 1 - mse / var if var > 0 else float("nan")}


def evaluate(model, dataset, loader, device, clients=None):
    """
    Metrics on a dataset with per-client breakdown.
    Returns {'all': metrics over `clients` rows (all rows if None), 0: ..., 1: ...}.
    """
    p = predict(model, loader, device)
    y, owner = dataset.labels, dataset.client
    sel = np.ones(len(y), bool) if clients is None else np.isin(owner, clients)
    out = {"all": regression_metrics(y[sel], p[sel])}
    for c in np.unique(owner):
        m = owner == c
        out[int(c)] = regression_metrics(y[m], p[m])
    return out


# ---------------------------------------------------------------- checkpoints / logs

def save_checkpoint(path, state):
    """Atomic save: a crash mid-write never leaves a corrupt checkpoint."""
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, device):
    # our checkpoints hold optimizer and RNG state, not just weights
    return torch.load(path, map_location=device, weights_only=False)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])


def prepare_run_dir(args, has_checkpoint=True):
    """
    Create the output directory and return the checkpoint to resume from (or None).
    Refuses to overwrite an existing run unless --resume is given, and refuses to resume
    with different hyperparameters. Use has_checkpoint=False when checkpoints live in subdirectories.
    """
    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    ckpt_path = os.path.join(out, "last.pt")
    cfg_path = os.path.join(out, "config.json")
    config = {k: v for k, v in vars(args).items()}

    if os.path.exists(ckpt_path) and not args.resume:
        raise SystemExit(f"{ckpt_path} exists. Pass --resume to continue it, or choose a new --output-dir.")

    if args.resume and os.path.exists(cfg_path):
        with open(cfg_path) as f:
            old = json.load(f)
        diff = {k: (old.get(k), config.get(k)) for k in set(old) | set(config)
                if k not in RESUMABLE_ARGS and old.get(k) != config.get(k)}
        if diff:
            raise SystemExit(f"Cannot resume {out} with different settings (old, new): {diff}")

    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    if args.resume and os.path.exists(ckpt_path):
        return ckpt_path
    if args.resume and has_checkpoint:
        print(f"No checkpoint in {out}; starting a new run.")
    return None


class MetricsCSV:
    """Append-only CSV. On resume, rows after the checkpointed step are dropped."""

    def __init__(self, path, step_key, keep_upto=None):
        self.path, self.step_key = path, step_key
        if keep_upto is None:
            if os.path.exists(path):
                os.remove(path)
        elif os.path.exists(path):
            with open(path, newline="") as f:
                rows = [r for r in csv.DictReader(f) if int(r[step_key]) <= keep_upto]
            os.remove(path)
            for r in rows:
                self.append(r)

    def append(self, row):
        new = not os.path.exists(self.path)
        if not new:
            with open(self.path, newline="") as f:
                fields = next(csv.reader(f))
        else:
            fields = list(row)
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)


def flat_metrics(prefix, metrics):
    return {f"{prefix}_{k}": (round(v, 6) if isinstance(v, float) else v)
            for k, v in metrics.items() if k != "n"}


def client_rows(step_key, step, split, results):
    """Long-format rows (one per client and one for 'all') for client_metrics.csv."""
    rows = []
    for c, m in results.items():
        rows.append({step_key: step, "split": split, "client": c, **{k: m[k] for k in ("n", "mse", "pearson", "ci", "r2")}})
    return rows


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)
