"""CLI entry point for training the honey-comb classifier."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    """Train the honey-comb classifier."""
    parser = argparse.ArgumentParser(
        description="Train the honey-comb classifier."
    )
    parser.add_argument(
        "train_path",
        help="Path to training JSONL file.",
    )
    parser.add_argument(
        "--eval",
        dest="eval_path",
        help="Path to evaluation JSONL file. If not provided, splits training data.",
    )
    parser.add_argument(
        "--output",
        default="models/honeycomb.joblib",
        help="Output path for trained model (default: models/honeycomb.joblib).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for evaluation when --eval is not provided (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    
    args = parser.parse_args()
    
    from honeycomb.classifier import train
    
    metrics = train(
        train_path=args.train_path,
        eval_path=args.eval_path,
        output_path=args.output,
        test_size=args.test_size,
        random_state=args.seed,
    )
    
    print(f"\nTraining complete. Accuracy: {metrics['accuracy']:.4f}")
    print(f"Model saved to: {metrics['output_path']}")


if __name__ == "__main__":
    main()
