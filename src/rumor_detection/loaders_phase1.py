#!/usr/bin/env python3
"""
Phase 1 — Dataset ingestion utilities and CLI

Goals
-----
1) Normalize different rumor/fake-news datasets into ONE schema:
      text,label,source,event_id,thread_id,doc_id
2) Provide simple commands to build clean CSVs you can train on.
3) Prevent leakage by optionally splitting by event_id (grouped split).
4) Offer a demo generator so you can test the pipeline with no raw data.

Supported loaders
-----------------
- PHEME directory structure (events/rumours/non-rumours/thread_id/...)
- FakeNewsNet directory structure (e.g., politifact_fake/real) with `news content.json`
- Generic CSV for Twitter/Reddit dumps (you specify text & label columns)

Example usage
-------------
# 1) Build from PHEME
python src/rumor_detection/loaders_phase1.py pheme-dir \
    --root data/raw/pheme --out data/pheme.csv

# 2) Build from FakeNewsNet
python src/rumor_detection/loaders_phase1.py fnn-dir \
    --root data/raw/fakenewsnet --out data/fnn.csv

# 3) Normalize your Twitter/Reddit CSVs
python src/rumor_detection/loaders_phase1.py social-csv \
    --in data/twitter.csv --text-col text --label-col label \
    --source Twitter --out data/twitter_norm.csv

# 4) Combine & deduplicate
python src/rumor_detection/loaders_phase1.py combine \
    --inputs data/pheme.csv data/fnn.csv data/twitter_norm.csv \
    --out data/combined.csv --dedup

# 5) Grouped split by event (prevents leakage)
python src/rumor_detection/loaders_phase1.py split \
    --data data/pheme.csv --group-col event_id --out-prefix data/pheme_split
# => data/pheme_split_train.csv, data/pheme_split_val.csv

# 6) Generate a tiny demo dataset (no raw data needed)
python src/rumor_detection/loaders_phase1.py demo --out data/demo.csv

Notes
-----
- Labels are normalized to {"Rumor", "Real"} via a mapping you can extend.
- For FakeNewsNet & PHEME, we try sensible defaults but these corpora vary; the
  loader is defensive and will skip files it cannot parse.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ----------------------------
# Common helpers
# ----------------------------

NORMAL_LABELS = {"Rumor", "Real"}

LABEL_MAPPING = {
    # General
    "rumor": "Rumor", "rumour": "Rumor", "fake": "Rumor", "false": "Rumor",
    "nonrumor": "Real", "non-rumor": "Real", "non-rumour": "Real", "real": "Real",
    "true": "Real", "legit": "Real", "reliable": "Real",
    # Numeric labels (added)
    "1": "Rumor", "1.0": "Rumor",
    "0": "Real",  "0.0": "Real",
    # PHEME folder names
    "rumours": "Rumor", "non-rumours": "Real",
}



def normalize_label(raw: str) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in LABEL_MAPPING:
        return LABEL_MAPPING[s]
    # common typos / variants
    s = s.replace(" ", "").replace("_", "-")
    if s in LABEL_MAPPING:
        return LABEL_MAPPING[s]
    # already normalized?
    if s.capitalize() in NORMAL_LABELS:
        return s.capitalize()
    return None


def clean_text_basic(t: str) -> str:
    t = str(t)
    t = re.sub(r"http\S+", " ", t)             # remove URLs
    t = re.sub(r"(?:^|\s)@\w+", " ", t)        # mentions (fixed)
    t = re.sub(r"#", " ", t)                   # hashtags → just remove #
    t = re.sub(r"\s+", " ", t).strip()         # extra spaces
    return t



def finalize_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    # Ensure essential columns
    needed = ["text", "label"]
    for col in needed:
        if col not in df.columns:
            df[col] = ""
    df["text"] = df["text"].astype(str).map(clean_text_basic)
    df["label"] = df["label"].map(normalize_label)
    df = df[df["label"].isin(NORMAL_LABELS)].copy()

    # Optional meta
    df["source"] = df.get("source", source_name)
    for opt in ["event_id", "thread_id", "doc_id"]:
        if opt not in df.columns:
            df[opt] = None

    # Drop empty texts
    df = df[df["text"].str.len() > 0]
    return df[["text", "label", "source", "event_id", "thread_id", "doc_id"]]


def print_stats(df: pd.DataFrame, name: str) -> None:
    print(f"Loaded {len(df):,} rows from {name}")
    if not df.empty:
        print("Label distribution:\n", df["label"].value_counts(normalize=True).round(3))


# ----------------------------
# Loader: Twitter/Reddit CSV
# ----------------------------

def load_social_csv(
    csv_path: str,
    text_col: str = "text",
    label_col: str = "label",
    source_name: str = "Social",
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"CSV must contain '{text_col}' and '{label_col}'. Found: {list(df.columns)}"
        )
    out = pd.DataFrame({
        "text": df[text_col].astype(str),
        "label": df[label_col],
        "source": source_name,
        "event_id": None,
        "thread_id": None,
        "doc_id": df.index.astype(str).map(lambda i: f"{source_name}:{i}"),
    })
    return finalize_df(out, source_name)


# ----------------------------
# Loader: FakeNewsNet directory
# ----------------------------

FNN_NEWS_JSON = {"news content.json", "news_content.json", "content.json"}


def _read_json_safely(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def _extract_text_from_fnn_json(obj: dict) -> str:
    parts: List[str] = []
    for k in ["title", "text", "content", "body", "main_text"]:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return "\n\n".join(parts)


def load_fakenewsnet_dir(root: str) -> pd.DataFrame:
    rootp = Path(root)
    rows = []
    if not rootp.exists():
        raise FileNotFoundError(f"FakeNewsNet root not found: {root}")

    # Expect subdirs like *fake* and *real*
    for label_dir in rootp.rglob("*"):
        if not label_dir.is_dir():
            continue
        name = label_dir.name.lower()
        if "fake" in name and "real" not in name:
            lab = "Rumor"
        elif "real" in name:
            lab = "Real"
        else:
            continue

        for art_dir in label_dir.iterdir():
            if not art_dir.is_dir():
                continue
            doc_id = str(art_dir.relative_to(rootp)).replace(os.sep, "/")
            text = ""
            # Look for a news JSON
            for fname in FNN_NEWS_JSON:
                p = art_dir / fname
                if p.exists():
                    obj = _read_json_safely(p)
                    if obj:
                        text = _extract_text_from_fnn_json(obj)
                        break
            # Fallback: any .txt
            if not text:
                txts = list(art_dir.glob("*.txt"))
                if txts:
                    try:
                        text = txts[0].read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        text = ""
            if text:
                rows.append({
                    "text": text,
                    "label": lab,
                    "source": "FakeNewsNet",
                    "event_id": None,
                    "thread_id": None,
                    "doc_id": doc_id,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        print("[WARN] FakeNewsNet loader found 0 articles. Check your directory structure.")
    return finalize_df(df, "FakeNewsNet")


# ----------------------------
# Loader: PHEME directory
# ----------------------------

"""
Expected layout (typical):
root/
  <event>/
    rumours/
      <thread_id>/
        source-tweet/<*.json>
        reactions/<*.json>  # optional to concatenate
    non-rumours/
      <thread_id>/
        source-tweet/<*.json>
        reactions/<*.json>
We will: take source-tweet text as the main item, optionally append first-level reactions.
"""


def _read_pheme_tweet_json(p: Path) -> Optional[str]:
    obj = _read_json_safely(p)
    if not obj:
        return None
    # Common PHEME keys
    for k in ["text", "tweet_text", "content"]:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v
    # If nested under 'data' (some dumps)
    data = obj.get("data") if isinstance(obj, dict) else None
    if isinstance(data, dict):
        v = data.get("text")
        if isinstance(v, str) and v.strip():
            return v
    return None


def load_pheme_dir(root: str, include_reactions: bool = False) -> pd.DataFrame:
    rootp = Path(root)
    rows = []
    if not rootp.exists():
        raise FileNotFoundError(f"PHEME root not found: {root}")

    for event_dir in sorted([d for d in rootp.iterdir() if d.is_dir()]):
        event_id = event_dir.name
        for cls_dir in [event_dir / "rumours", event_dir / "non-rumours"]:
            if not cls_dir.exists():
                continue
            lab = "Rumor" if cls_dir.name == "rumours" else "Real"

            for thread_dir in [d for d in cls_dir.iterdir() if d.is_dir()]:
                thread_id = thread_dir.name
                src_dir = thread_dir / "source-tweet"
                if not src_dir.exists():
                    continue
                src_jsons = list(src_dir.glob("*.json"))
                if not src_jsons:
                    continue
                text_main = _read_pheme_tweet_json(src_jsons[0]) or ""

                # Optionally add reactions (concatenated)
                if include_reactions:
                    rx_dir = thread_dir / "reactions"
                    if rx_dir.exists():
                        parts: List[str] = [text_main]
                        for rx in rx_dir.glob("*.json"):
                            t = _read_pheme_tweet_json(rx)
                            if t:
                                parts.append(t)
                        text_main = "\n\n".join(parts)

                if text_main.strip():
                    rows.append({
                        "text": text_main,
                        "label": lab,
                        "source": "PHEME",
                        "event_id": event_id,
                        "thread_id": thread_id,
                        "doc_id": f"{event_id}/{thread_id}",
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        print("[WARN] PHEME loader found 0 items. Check your directory structure.")
    return finalize_df(df, "PHEME")


# ----------------------------
# Combine & deduplicate & split
# ----------------------------

def combine_and_dedup(inputs: List[str], out_path: str, dedup: bool = True) -> pd.DataFrame:
    frames = []
    for p in inputs:
        if not os.path.exists(p):
            print(f"[WARN] Missing input: {p}")
            continue
        frames.append(pd.read_csv(p))
    if not frames:
        raise ValueError("No valid input CSVs found.")
    df = pd.concat(frames, ignore_index=True)

    # Basic cleaning for safety
    df["text"] = df["text"].astype(str)

    if dedup:
        key = df["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        before = len(df)
        df = df.loc[~key.duplicated()].copy()
        after = len(df)
        print(f"Deduplicated: {before-after} rows removed")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print_stats(df, f"combined -> {out_path}")
    return df


def grouped_split(
    data_csv: str,
    out_prefix: str,
    group_col: str = "event_id",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(data_csv)
    if group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not in CSV columns: {list(df.columns)}")

    # Remove rows with missing group
    df = df[~df[group_col].isna()].copy()
    groups = df[group_col].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    (train_idx, val_idx) = next(splitter.split(df, groups=groups))
    train_df, val_df = df.iloc[train_idx].copy(), df.iloc[val_idx].copy()

    base = os.path.dirname(out_prefix)
    os.makedirs(base or ".", exist_ok=True)
    train_path = f"{out_prefix}_train.csv"
    val_path = f"{out_prefix}_val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    print_stats(train_df, train_path)
    print_stats(val_df, val_path)
    return train_df, val_df


# ----------------------------
# Demo generator
# ----------------------------

def generate_demo(out_path: str) -> pd.DataFrame:
    demo = pd.DataFrame([
        {"text": "Breaking: Government bans sleep on weekdays", "label": "Rumor", "source": "Demo"},
        {"text": "NASA confirms water on the Moon's south pole", "label": "Real", "source": "Demo"},
        {"text": "Cure for common cold found in local market", "label": "Rumor", "source": "Demo"},
        {"text": "WHO releases new vaccination guidelines", "label": "Real", "source": "Demo"},
        {"text": "Celebrity adopts 1000 cats overnight", "label": "Rumor", "source": "Demo"},
        {"text": "New study links exercise to longevity", "label": "Real", "source": "Demo"},
    ])
    demo["event_id"] = None
    demo["thread_id"] = None
    demo["doc_id"] = [f"demo:{i}" for i in range(len(demo))]

    demo = finalize_df(demo, "Demo")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    demo.to_csv(out_path, index=False)
    print_stats(demo, f"demo -> {out_path}")
    return demo


# ----------------------------
# CLI
# ----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 1 data ingestion CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # social-csv
    s = sub.add_parser("social-csv", help="Normalize a Twitter/Reddit CSV")
    s.add_argument("--in", dest="csv_in", required=True)
    s.add_argument("--text-col", default="text")
    s.add_argument("--label-col", default="label")
    s.add_argument("--source", default="Social")
    s.add_argument("--out", required=True)

    # fakenewsnet dir
    f = sub.add_parser("fnn-dir", help="Parse a FakeNewsNet directory")
    f.add_argument("--root", required=True)
    f.add_argument("--out", required=True)

    # pheme dir
    ph = sub.add_parser("pheme-dir", help="Parse a PHEME directory")
    ph.add_argument("--root", required=True)
    ph.add_argument("--out", required=True)
    ph.add_argument("--include-reactions", action="store_true")

    # combine
    c = sub.add_parser("combine", help="Combine multiple normalized CSVs")
    c.add_argument("--inputs", nargs="+", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--dedup", action="store_true")

    # split
    sp = sub.add_parser("split", help="Grouped train/val split (e.g., by event)")
    sp.add_argument("--data", required=True)
    sp.add_argument("--out-prefix", required=True)
    sp.add_argument("--group-col", default="event_id")
    sp.add_argument("--test-size", type=float, default=0.2)

    # demo
    d = sub.add_parser("demo", help="Generate a tiny demo CSV (no raw data)")
    d.add_argument("--out", required=True)

    return p


def main():
    args = build_arg_parser().parse_args()

    if args.cmd == "social-csv":
        df = load_social_csv(args.csv_in, args.text_col, args.label_col, args.source)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df.to_csv(args.out, index=False)
        print_stats(df, args.out)

    elif args.cmd == "fnn-dir":
        df = load_fakenewsnet_dir(args.root)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df.to_csv(args.out, index=False)
        print_stats(df, args.out)

    elif args.cmd == "pheme-dir":
        df = load_pheme_dir(args.root, include_reactions=args.include_reactions)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df.to_csv(args.out, index=False)
        print_stats(df, args.out)

    elif args.cmd == "combine":
        combine_and_dedup(args.inputs, args.out, dedup=args.dedup)

    elif args.cmd == "split":
        grouped_split(args.data, args.out_prefix, group_col=args.group_col, test_size=args.test_size)

    elif args.cmd == "demo":
        generate_demo(args.out)

    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
