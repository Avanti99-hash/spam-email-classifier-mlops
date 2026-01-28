from __future__ import annotations

import argparse
from pathlib import Path

from .ingest import ingest
from .preprocess import build_preprocessed_df, bootstrap
from .train import train_pipeline
from .evaluate import run_evaluation

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--zip", required=True, help="Path to Combined_Spam_Ham.zip")
    t.add_argument("--extract-dir", default="data/extracted", help="Where to extract zip")
    t.add_argument("--artifacts-dir", default="artifacts")
    t.add_argument("--rows", type=int, default=15000)
    t.add_argument("--min-accuracy", type=float, default=0.90)

    args = p.parse_args()

    if args.cmd == "train":
        zip_path = Path(args.zip)
        extract_dir = Path(args.extract_dir)
        artifacts_dir = Path(args.artifacts_dir)

        email_df = ingest(zip_path, extract_dir)
        pre = build_preprocessed_df(email_df)
        boot = bootstrap(pre, desired_rows=args.rows)

        X_train, X_test, y_train, y_test = train_pipeline(boot, artifacts_dir)
        payload = run_evaluation(
            artifacts_dir,
            X_train, y_train,
            X_test, y_test,
            min_accuracy=args.min_accuracy,
        )
        print("Training complete. Best:", payload["best_model"], payload["best_metrics"])


if __name__ == "__main__":
    main()
