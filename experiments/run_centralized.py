"""
Centralized baseline: one MPNN-CNN model trained on the union of all client training sets.

Example:
    python experiments/run_centralized.py --epochs 100 --output-dir outputs/centralized --resume
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.engine import add_common_args, get_device, set_seed, prepare_run_dir
from utils.features import featurize_protocol
from utils.trainer import train_supervised


def main(argv=None):
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args(argv)

    ckpt = prepare_run_dir(args)
    device = get_device(args.device)
    set_seed(args.seed)
    print(f"Centralized training on {device}, output: {args.output_dir}")

    data = featurize_protocol(num_clients=args.num_clients, seed=args.split_seed, subsample=args.subsample)
    g = data["global"]
    print(f"train {len(g['train'])}  val {len(g['val'])}  test {len(g['test'])}")
    return train_supervised(args, g["train"], g["val"], g["test"], args.output_dir, device, ckpt_path=ckpt)


if __name__ == "__main__":
    main()
