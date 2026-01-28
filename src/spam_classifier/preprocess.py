from __future__ import annotations

import re

import pandas as pd
from nltk.corpus import stopwords


def clean_text(text: str) -> str:
    if text is None:
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove non-alphanumeric characters (except spaces)
    text = re.sub(r"[^\w\s]", " ", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)
    # Remove email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Remove system tokens
    text = re.sub(
        r"\b(html|body|font|color|size|center|pythonnet|cbgovernmentgw|gward)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Lowercase
    text = text.lower()

    stop_words = set(stopwords.words("english"))
    words = text.split()
    filtered_words = [w for w in words if w not in stop_words]
    return " ".join(filtered_words).strip()


def build_preprocessed_df(email_df: pd.DataFrame) -> pd.DataFrame:
    df = email_df.copy()
    df["Cleaned_Body"] = df["Body"].astype(str).apply(clean_text)
    out = df[["FileName", "Sender", "Receiver", "Subject", "Cleaned_Body"]].copy()
    return out


def bootstrap(df: pd.DataFrame, desired_rows: int = 15000, seed: int = 42) -> pd.DataFrame:
    boot = df.sample(n=desired_rows, replace=True, random_state=seed).reset_index(drop=True)
    boot = boot.dropna(subset=["Sender", "Subject", "Cleaned_Body"]).reset_index(drop=True)
    boot["Combined_Text"] = (boot["Sender"].astype(str) + " " +
                            boot["Subject"].astype(str) + " " +
                            boot["Cleaned_Body"].astype(str))
    boot["Combined_Text"] = boot["Combined_Text"].astype(str)
    return boot
