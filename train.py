"""CLI: python train.py [--model EfficientNetV2B0|all] [--epochs N] [--subset N]"""
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train_lib import run_training, compare_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EfficientNetV2B0")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--data", default=None)
    ap.add_argument("--subset", type=int, default=None,
                    help="cap images per split (smoke tests)")
    a = ap.parse_args()
    if a.model == "all":
        compare_all(epochs=a.epochs, batch=a.batch,
                    data_dir=a.data, subset=a.subset)
    else:
        run_training(a.model, epochs=a.epochs, batch=a.batch,
                     data_dir=a.data, subset=a.subset)


if __name__ == "__main__":
    main()
