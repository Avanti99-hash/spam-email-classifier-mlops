from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class IngestConfig:
    zip_path: Path
    extract_dir: Path


def extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    return extract_dir


def _find_email_folder(extract_dir: Path) -> Path:
    """
    Your notebook assumes: extracted_folder/Combined_Spam_Ham/<email files>
    We'll search for a folder that contains lots of files.
    """
    candidates = []
    for root, dirs, files in os.walk(extract_dir):
        if files:
            candidates.append((root, len(files)))
    if not candidates:
        raise FileNotFoundError(f"No files found after extracting {extract_dir}")
    candidates.sort(key=lambda x: x[1], reverse=True)
    return Path(candidates[0][0])


def parse_email_files(email_folder: Path) -> pd.DataFrame:
    email_data = []
    for email_file in sorted(email_folder.iterdir()):
        if not email_file.is_file():
            continue
        try:
            content = email_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
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

    return pd.DataFrame(email_data)


def ingest(zip_path: Path, extract_dir: Path) -> pd.DataFrame:
    extract_zip(zip_path, extract_dir)
    email_folder = _find_email_folder(extract_dir)
    df = parse_email_files(email_folder)
    return df
