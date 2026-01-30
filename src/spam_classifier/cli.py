from __future__ import annotations

import argparse
from pathlib import Path
import logging
import time

from .logging_config import setup_logging
from .ingest import ingest
from .preprocess import build_preprocessed_df, bootstrap
from .train import train_pipeline
from .evaluate import run_evaluation
logger = logging.getLogger(__name__)

def main():
    setup_logging(app_name="spam_classifier", log_file="spam_classifier_cli.log", always_to_file=True)
    logger.info("CLI started")
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--zip", required=True, help="Path to Combined_Spam_Ham.zip")
    t.add_argument("--extract-dir", default="data/extracted", help="Where to extract zip")
    t.add_argument("--artifacts-dir", default="artifacts")
    t.add_argument("--rows", type=int, default=15000)
    t.add_argument("--min-accuracy", type=float, default=0.90)

    args = p.parse_args()
    logger.info("Command: %s", args.cmd)

    if args.cmd == "train":
        try:
            zip_path = Path(args.zip)
            extract_dir = Path(args.extract_dir)
            artifacts_dir = Path(args.artifacts_dir)
            logger.info("Train config: zip=%s extract_dir=%s artifacts_dir=%s rows=%d min_accuracy=%.2f",
                zip_path, extract_dir, artifacts_dir, args.rows, args.min_accuracy)

            t0 = time.time()
            logger.info("Stage: ingest started")
            email_df = ingest(zip_path, extract_dir)
            logger.info("Stage: ingest completed in %.2fs | rows=%d", time.time() - t0, len(email_df))

            t0 = time.time()
            logger.info("Stage: preprocess started")
            pre = build_preprocessed_df(email_df)
            logger.info("Stage: preprocess completed in %.2fs | rows=%d", time.time() - t0, len(pre))

            t0 = time.time()
            logger.info("Stage: bootstrap started")
            boot = bootstrap(pre, desired_rows=args.rows)
            logger.info("Stage: bootstrap completed in %.2fs | rows=%d", time.time() - t0, len(boot))

            t0 = time.time()
            logger.info("Stage: train started")
            X_train, X_test, y_train, y_test = train_pipeline(boot, artifacts_dir)
            logger.info("Stage: train completed in %.2fs | X_train=%s X_test=%s y_train=%d y_test=%d",
                time.time() - t0, X_train.shape, X_test.shape, len(y_train), len(y_test))

            t0 = time.time()
            logger.info("Stage: evaluate started (quality gate min_accuracy=%.2f)", args.min_accuracy)
            payload = run_evaluation(
            artifacts_dir,
            X_train, y_train,
            X_test, y_test,
            min_accuracy=args.min_accuracy,
            )
            logger.info("Stage: evaluate completed in %.2fs | best_model=%s | best_acc=%.4f",
                time.time() - t0, payload["best_model"], payload["best_metrics"]["accuracy"])

            print("Training complete. Best:", payload["best_model"], payload["best_metrics"])

        except Exception:
            logger.exception("Training pipeline failed")
            raise

if __name__ == "__main__":
    main()
