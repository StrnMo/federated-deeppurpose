"""
Federated learning with FedAvg across the non-IID (drug-partitioned) clients.

Example:
    python experiments/run_federated.py --rounds 30 --local-epochs 2 --output-dir outputs/fedavg --resume
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.engine import add_common_args, get_device, set_seed, prepare_run_dir
from utils.features import featurize_protocol
from federated.fedavg import run_fedavg


def main(argv=None):
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=2)
    args = parser.parse_args(argv)

    ckpt = prepare_run_dir(args)
    device = get_device(args.device)
    set_seed(args.seed)
    print(f"FedAvg on {device}: {args.rounds} rounds x {args.local_epochs} local epochs, output: {args.output_dir}")

    data = featurize_protocol(num_clients=args.num_clients, seed=args.split_seed, subsample=args.subsample)
    for k, d in enumerate(data["clients"]):
        print(f"client {k}: train {len(d['train'])}")
    return run_fedavg(args, data, device, args.output_dir, ckpt_path=ckpt)


if __name__ == "__main__":
    main()
