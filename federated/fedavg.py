"""
FedAvg communication rounds with checkpointing, resume and CSV metrics.

Each round: broadcast global weights -> every client trains locally -> sample-weighted
aggregation -> evaluate the global model on the global val and test sets.
The best round is selected on global validation MSE.

Run directory layout matches utils/trainer.py, with `round` in place of `epoch`.
"""

import os
import time

from federated.client import FLClient
from federated.server import FLServer
from utils.features import DTIDataset
from utils.engine import (build_model, evaluate, save_checkpoint, load_checkpoint, rng_state, set_rng_state,
                          MetricsCSV, flat_metrics, client_rows)
from utils.trainer import cpu_state, eval_loader, finalize


def run_fedavg(args, data, device, out_dir, ckpt_path=None):
    clients = [FLClient(k, d["train"]) for k, d in enumerate(data["clients"])]
    val_ds, test_ds = DTIDataset(data["global"]["val"]), DTIDataset(data["global"]["test"])
    sizes = [c.num_samples for c in clients]

    model = build_model(device, args.seed)  # same initial weights for every client
    server = FLServer(cpu_state(model))
    start, best = 1, {"val_mse": float("inf"), "round": 0, "state": None}

    if ckpt_path:
        ck = load_checkpoint(ckpt_path, "cpu")
        server.global_weights = ck["global"]
        set_rng_state(ck["rng"])
        start, best = ck["round"] + 1, ck["best"]
        print(f"Resumed from round {ck['round']} (best round {best['round']})")

    keep = start - 1 if ckpt_path else None
    metrics_csv = MetricsCSV(os.path.join(out_dir, "metrics.csv"), "round", keep)
    client_csv = MetricsCSV(os.path.join(out_dir, "client_metrics.csv"), "round", keep)

    for rnd in range(start, args.rounds + 1):
        t0 = time.time()
        global_weights = server.get_global_weights()

        updates, losses = [], []
        for c in clients:
            seed = args.seed * 100003 + rnd * 1009 + c.client_id * 101
            w, loss = c.train(model, global_weights, args.local_epochs, args.lr, args.batch_size, device, seed,
                              args.num_workers)
            updates.append(w)
            losses.append(loss)

        server.aggregate(updates, sizes)
        model.load_state_dict(server.get_global_weights())
        train_loss = sum(l * n for l, n in zip(losses, sizes)) / sum(sizes)

        val = evaluate(model, val_ds, eval_loader(val_ds, args), device)
        test = evaluate(model, test_ds, eval_loader(test_ds, args), device)

        improved = val["all"]["mse"] < best["val_mse"]
        if improved:
            best = {"val_mse": val["all"]["mse"], "round": rnd, "state": server.get_global_weights()}

        metrics_csv.append({"round": rnd, "train_loss": round(train_loss, 6), **flat_metrics("val", val["all"]),
                            **flat_metrics("test", test["all"]), "best_round": best["round"],
                            "seconds": round(time.time() - t0, 1)})
        for row in client_rows("round", rnd, "val", val) + client_rows("round", rnd, "test", test):
            client_csv.append(row)

        save_checkpoint(os.path.join(out_dir, "last.pt"), {"round": rnd, "global": server.get_global_weights(),
                        "best": best, "rng": rng_state()})
        if improved:
            save_checkpoint(os.path.join(out_dir, "best.pt"), {"round": rnd, "model": best["state"]})

        print(f"round {rnd}/{args.rounds}  train {train_loss:.4f}  val MSE {val['all']['mse']:.4f} "
              f"CI {val['all']['ci']:.4f}  test MSE {test['all']['mse']:.4f} CI {test['all']['ci']:.4f}  "
              f"{time.time() - t0:.0f}s{'  *' if improved else ''}")

    return finalize(model, best, val_ds, test_ds, args, device, out_dir, "round")
