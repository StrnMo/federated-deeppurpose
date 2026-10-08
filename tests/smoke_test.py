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

        print("SMOKE TEST PASSED")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
