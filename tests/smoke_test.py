"""
Quick CPU end-to-end test of the centralized, local-only and FedAvg scripts on a tiny subsample,
including checkpoint/resume. Takes a few minutes on CPU.

    python tests/smoke_test.py
"""

import os
import sys
import json
import shutil
import tempfile
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import run_centralized, run_local, run_federated

COMMON = ["--subsample", "0.02", "--device", "cpu", "--batch-size", "32"]


def expect_exit(fn, argv, text):
    try:
        fn(argv)
    except SystemExit as e:
        assert text in str(e), f"unexpected message: {e}"
        return
    raise AssertionError(f"expected SystemExit containing '{text}'")


def check_run(out, step, n_steps, n_clients=5, files=("last.pt", "best.pt", "config.json", "final_metrics.json")):
    m = pd.read_csv(os.path.join(out, "metrics.csv"))
    assert list(m[step]) == list(range(1, n_steps + 1)), f"{out}: steps {list(m[step])}"
    cm = pd.read_csv(os.path.join(out, "client_metrics.csv"))
    assert len(cm) == n_steps * 2 * (n_clients + 1), f"{out}: {len(cm)} client rows"
    for f in files:
        assert os.path.exists(os.path.join(out, f)), f"missing {f}"
    with open(os.path.join(out, "final_metrics.json")) as f:
        return json.load(f)


def check_early_stopping(root):
    """
    With --lr 0 the weights never change, so val MSE is identical every step: step 1 stays best and
    early stopping must trigger exactly `patience` steps later, also across resumes.
    """
    def final(out):
        with open(os.path.join(out, "final_metrics.json")) as f:
            return json.load(f)

    def steps(out, key):
        return list(pd.read_csv(os.path.join(out, "metrics.csv"))[key])

    lr0 = COMMON + ["--lr", "0"]

    # centralized: patience 1 -> epochs 1, 2 then stop; resuming a stopped run trains nothing more
    out = os.path.join(root, "es_central")
    run_centralized.main(lr0 + ["--epochs", "5", "--patience", "1", "--output-dir", out])
    assert steps(out, "epoch") == [1, 2], steps(out, "epoch")
    run_centralized.main(lr0 + ["--epochs", "8", "--patience", "1", "--output-dir", out, "--resume"])
    assert steps(out, "epoch") == [1, 2], steps(out, "epoch")
    f = final(out)
    assert f["stopped_early"] and f["best_epoch"] == 1, f
    m = pd.read_csv(os.path.join(out, "metrics.csv"))
    assert "val_bias" in m.columns and "test_bias" in m.columns
    # the reported model is the best one (epoch 1)
    assert abs(f["test"]["all"]["mse"] - m.loc[0, "test_mse"]) < 1e-5

    # centralized, interrupted before stopping: 2 epochs, then resume -> epochs 3, 4, stop before 5
    out = os.path.join(root, "es_central_mid")
    run_centralized.main(lr0 + ["--epochs", "2", "--patience", "3", "--output-dir", out])
    assert not final(out)["stopped_early"]
    run_centralized.main(lr0 + ["--epochs", "10", "--patience", "3", "--output-dir", out, "--resume"])
    assert steps(out, "epoch") == [1, 2, 3, 4], steps(out, "epoch")
    assert final(out)["stopped_early"] and final(out)["best_epoch"] == 1

    # FedAvg: 2 rounds, resume -> round 3, stop before round 4 (patience 2)
    out = os.path.join(root, "es_fedavg")
    run_federated.main(lr0 + ["--rounds", "2", "--local-epochs", "1", "--patience", "2", "--output-dir", out])
    run_federated.main(lr0 + ["--rounds", "10", "--local-epochs", "1", "--patience", "2", "--output-dir", out,
                              "--resume"])
    assert steps(out, "round") == [1, 2, 3], steps(out, "round")
    assert final(out)["stopped_early"] and final(out)["best_round"] == 1


def main():
    root = tempfile.mkdtemp(prefix="fl_smoke_")
    try:
        # centralized: 2 epochs, then resume to 3
        out = os.path.join(root, "centralized")
        run_centralized.main(COMMON + ["--epochs", "2", "--output-dir", out])
        expect_exit(run_centralized.main, COMMON + ["--epochs", "3", "--output-dir", out], "--resume")
        expect_exit(run_centralized.main, COMMON + ["--epochs", "3", "--lr", "0.01", "--output-dir", out, "--resume"],
                    "different settings")
        run_centralized.main(COMMON + ["--epochs", "3", "--output-dir", out, "--resume"])
        check_run(out, "epoch", 3)
        print("\n[ok] centralized + resume\n")

        # local-only: 1 epoch per client, then resume to 2
        out = os.path.join(root, "local")
        run_local.main(COMMON + ["--epochs", "1", "--output-dir", out])
        run_local.main(COMMON + ["--epochs", "2", "--output-dir", out, "--resume"])
        for k in range(5):
            check_run(os.path.join(out, f"client_{k}"), "epoch", 2, files=("last.pt", "best.pt", "final_metrics.json"))
        assert os.path.exists(os.path.join(out, "config.json"))
        assert len(pd.read_csv(os.path.join(out, "local_summary.csv"))) == 5
        print("\n[ok] local-only + resume\n")

        # FedAvg: 2 rounds, then resume to 3
        out = os.path.join(root, "fedavg")
        run_federated.main(COMMON + ["--rounds", "2", "--local-epochs", "1", "--output-dir", out])
        run_federated.main(COMMON + ["--rounds", "3", "--local-epochs", "1", "--output-dir", out, "--resume"])
        check_run(out, "round", 3)
        print("\n[ok] FedAvg + resume\n")

        check_early_stopping(root)
        print("\n[ok] early stopping + resume\n")

        print("SMOKE TEST PASSED")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
