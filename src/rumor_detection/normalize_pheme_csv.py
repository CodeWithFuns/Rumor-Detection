import argparse
import pandas as pd
import hashlib
from pathlib import Path
from rumor_detection.preprocess import clean_text

LABEL_MAP = {1.0: "rumor", 1: "rumor", "1": "rumor",
             0.0: "non-rumor", 0: "non-rumor", "0": "non-rumor"}

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def normalize_pheme_csv(csv_in: str, csv_out: str, drop_na: bool = True, dedup: bool = True):
    df = pd.read_csv(csv_in)
    required = {"text", "is_rumor", "topic"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {csv_in}: {missing} (have={list(df.columns)})")

    # map labels
    df["label"] = df["is_rumor"].map(LABEL_MAP)
    if drop_na:
        df = df[df["label"].notna()].copy()

    # unified schema
    df["source"] = "PHEME"
    df["event_id"] = df["topic"].astype(str)
    df["thread_id"] = ""  # not available in this CSV
    df["doc_id"] = df["text"].astype(str).map(clean_text).map(sha1_text)

    out_cols = ["text", "label", "source", "event_id", "thread_id", "doc_id"]
    df_out = df[out_cols].copy()

    if dedup:
        before = len(df_out)
        df_out = df_out.drop_duplicates(subset=["doc_id"], keep="first")
        after = len(df_out)
        print(f"Deduped by doc_id: {before} -> {after} rows")

    Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(csv_out, index=False)
    print(f"Wrote normalized PHEME CSV to {csv_out} ({len(df_out)} rows)")

def parse_args():
    ap = argparse.ArgumentParser(description="Normalize PHEME CSV to unified schema")
    ap.add_argument("--in", dest="csv_in", default="data/pheme/dataset.csv")
    ap.add_argument("--out", dest="csv_out", default="data/pheme/pheme_norm.csv")
    ap.add_argument("--keep-na", action="store_true", help="Keep rows with unknown labels")
    ap.add_argument("--no-dedup", action="store_true", help="Do not deduplicate on doc_id")
    return ap.parse_args()

def main():
    args = parse_args()
    normalize_pheme_csv(args.csv_in, args.csv_out, drop_na=not args.keep_na, dedup=not args.no_dedup)

if __name__ == "__main__":
    main()
