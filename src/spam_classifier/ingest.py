from __future__ import annotations

import os
import re
import zipfile
import logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import logging
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class IngestConfig:
    zip_path: Path
    extract_dir: Path


def extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    logger.info("Extracting zip file: %s -> %s", zip_path, extract_dir)

    if not zip_path.exists():
        logger.error("Zip file does not exist: %s", zip_path)
        raise FileNotFoundError(zip_path)

    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    logger.info("Zip extraction completed")
    return extract_dir


def _find_email_folder(extract_dir: Path) -> Path:
    logger.info("Searching for email files under: %s", extract_dir)

    candidates = []
    for root, dirs, files in os.walk(extract_dir):
        if files:
            candidates.append((root, len(files)))

    if not candidates:
        logger.error("No files found after extracting %s", extract_dir)
        raise FileNotFoundError(f"No files found after extracting {extract_dir}")

    candidates.sort(key=lambda x: x[1], reverse=True)
    chosen = Path(candidates[0][0])

    logger.info(
        "Detected email folder: %s (files=%d)",
        chosen,
        candidates[0][1],
    )

    return chosen

def parse_email_files(email_folder: Path) -> pd.DataFrame:
    logger.info("Parsing email files from folder: %s", email_folder)
    email_data = []
    for email_file in sorted(email_folder.iterdir()):
        if not email_file.is_file():
            continue
        try:
            content = email_file.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.warning("Skipping unreadable file: %s | error=%s", email_file.name, e)
            continue

        sender = re.search(r"From: (.*)", content)
        receiver = re.search(r"To: (.*)", content)
        subject = re.search(r"Subject: (.*)", content)

        body_start = content.find("\n\n")
        body = content[body_start:].strip() if body_start != -1 else ""

        email_data.append(
            {
                "FileName": email_file.name,
                "Sender": sender.group(1).strip() if sender else None,
                "Receiver": receiver.group(1).strip() if receiver else None,
                "Subject": subject.group(1).strip() if subject else None,
                "Body": body,
            }
        )

        if len(email_data) % 500 == 0:
            logger.info("Parsed %d emails so far...", len(email_data))

    logger.info("Completed parsing emails | total_parsed=%d", len(email_data))
    return pd.DataFrame(email_data)


def ingest(zip_path: Path, extract_dir: Path) -> pd.DataFrame:
    logger.info("Ingest pipeline started")

    extract_zip(zip_path, extract_dir)
    email_folder = _find_email_folder(extract_dir)
    df = parse_email_files(email_folder)

    logger.info("Ingest pipeline completed | rows=%d", len(df))
    return df
