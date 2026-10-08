"""
Epoch loop with checkpointing, resume and CSV metrics for the centralized and
local-only baselines, plus the final evaluation shared with FedAvg.

Run directory layout:
    config.json          arguments of the run
    last.pt              latest epoch: model, optimizer, best model, RNG state
    best.pt              model with the lowest validation MSE so far
    metrics.csv          one row per epoch (val_* = model-selection set, test_* = global test)
    client_metrics.csv   per-client val/test metrics per epoch (long format)
    final_metrics.json   best model evaluated on val and test (written when training ends)
"""

import os
import time
import torch

from utils.features import DTIDataset, make_loader
from utils.engine import (build_model, train_one_epoch, evaluate, save_checkpoint, load_checkpoint,
                          rng_state, set_rng_state, MetricsCSV, flat_metrics, client_rows, write_json)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def eval_loader(dataset, args):
    return make_loader(dataset, args.batch_size, num_workers=args.num_workers)


def finalize(model, best, val_ds, test_ds, args, device, out_dir, step_key, select_clients=None):
    """Evaluate the best model on val and test and write final_metrics.json."""
    model.load_state_dict(best["state"])
    val = evaluate(model, val_ds, eval_loader(val_ds, args), device, clients=select_clients)
    test = evaluate(model, test_ds, eval_loader(test_ds, args), device)
    result = {f"best_{step_key}": best[step_key], "selection_clients": select_clients,
              "val": {str(k): v for k, v in val.items()}, "test": {str(k): v for k, v in test.items()}}
    write_json(os.path.join(out_dir, "final_metrics.json"), result)
    t = test["all"]
    print(f"Best {step_key} {best[step_key]}: test MSE {t['mse']:.4f}  Pearson {t['pearson']:.4f}  "
          f"CI {t['ci']:.4f}  R2 {t['r2']:.4f}")
    return result


def train_supervised(args, train_df, val_df, test_df, out_dir, device, select_clients=None, ckpt_path=None):
    """
    Train one model on train_df for args.epochs epochs.
    Model selection uses val rows of `select_clients` (all rows if None); test is always reported in full.
    """
    os.makedirs(out_dir, exist_ok=True)
    train_ds, val_ds, test_ds = DTIDataset(train_df), DTIDataset(val_df), DTIDataset(test_df)

    label_mean = float(train_ds.labels.mean()) if args.init_bias == "label_mean" else None
    model = build_model(device, args.seed, label_mean)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    start, best = 1, {"val_mse": float("inf"), "epoch": 0, "state": None}

    if ckpt_path:
        ck = load_checkpoint(ckpt_path, device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        set_rng_state(ck["rng"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"Resumed from epoch {ck['epoch']} (best epoch {best['epoch']})")

    keep = start - 1 if ckpt_path else None
    metrics_csv = MetricsCSV(os.path.join(out_dir, "metrics.csv"), "epoch", keep)
    client_csv = MetricsCSV(os.path.join(out_dir, "client_metrics.csv"), "epoch", keep)

    for epoch in range(start, args.epochs + 1):
        t0 = time.time()
        loader = make_loader(train_ds, args.batch_size, shuffle=True, seed=args.seed * 100003 + epoch,
                             num_workers=args.num_workers)
        train_loss = train_one_epoch(model, loader, opt, device)
        val = evaluate(model, val_ds, eval_loader(val_ds, args), device, clients=select_clients)
        test = evaluate(model, test_ds, eval_loader(test_ds, args), device)

        improved = val["all"]["mse"] < best["val_mse"]
        if improved:
            best = {"val_mse": val["all"]["mse"], "epoch": epoch, "state": cpu_state(model)}

        metrics_csv.append({"epoch": epoch, "train_loss": round(train_loss, 6), **flat_metrics("val", val["all"]),
                            **flat_metrics("test", test["all"]), "best_epoch": best["epoch"],
                            "seconds": round(time.time() - t0, 1)})
        for row in client_rows("epoch", epoch, "val", val) + client_rows("epoch", epoch, "test", test):
            client_csv.append(row)

        save_checkpoint(os.path.join(out_dir, "last.pt"), {"epoch": epoch, "model": model.state_dict(),
                        "optimizer": opt.state_dict(), "best": best, "rng": rng_state()})
        if improved:
            save_checkpoint(os.path.join(out_dir, "best.pt"), {"epoch": epoch, "model": best["state"]})

        print(f"epoch {epoch}/{args.epochs}  train {train_loss:.4f}  val MSE {val['all']['mse']:.4f} "
              f"CI {val['all']['ci']:.4f}  test MSE {test['all']['mse']:.4f} CI {test['all']['ci']:.4f}  "
              f"{time.time() - t0:.0f}s{'  *' if improved else ''}")

    return finalize(model, best, val_ds, test_ds, args, device, out_dir, "epoch", select_clients)
