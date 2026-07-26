"""Converts the raw Kaggle "Phishing Email Detection" CSV
(subhajournal/phishingemails — columns: Email Text, Email Type) into the
prompt/completion JSONL format expected by mlx_lm.lora, with a stratified
train/valid split.

Cleaning steps (checked, not assumed — see PROGRESS.md 2026-07-26 analysis):
  - drop rows with empty Email Text (19 in the raw file)
  - drop duplicate Email Text rows (1112 in the raw file) to avoid the same
    email landing in both train and valid (data leakage)
  - drop rows over --max-length chars (default 5000): the raw file has a
    long tail up to 17M chars against a median of 880; such rows would
    dominate fine-tuning context for no benefit
  - split is stratified by Email Type so train/valid keep the same
    Safe/Phishing ratio as the source data
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

LABEL_MAP = {
    "Safe Email": "Safe",
    "Phishing Email": "Phishing",
}


def load_rows(csv_path, max_length):
    seen_text = set()
    rows = []
    stats = {"total": 0, "empty": 0, "duplicate": 0, "too_long": 0, "kept": 0}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["total"] += 1
            text = (row.get("Email Text") or "").strip()
            label = row.get("Email Type")

            if not text:
                stats["empty"] += 1
                continue
            if text in seen_text:
                stats["duplicate"] += 1
                continue
            if len(text) > max_length:
                stats["too_long"] += 1
                continue

            seen_text.add(text)
            rows.append({"prompt": text, "completion": LABEL_MAP.get(label, label)})
            stats["kept"] += 1

    return rows, stats


def stratified_split(rows, valid_fraction, seed):
    by_label = {}
    for row in rows:
        by_label.setdefault(row["completion"], []).append(row)

    rng = random.Random(seed)
    train, valid = [], []
    for label, group in by_label.items():
        rng.shuffle(group)
        n_valid = round(len(group) * valid_fraction)
        valid.extend(group[:n_valid])
        train.extend(group[n_valid:])

    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="outputs/kaggle_raw/Phishing_Email.csv")
    parser.add_argument("--max-length", type=int, default=5000, help="Drop Email Text longer than this (chars).")
    parser.add_argument("--valid-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-out", default="data/train.jsonl")
    parser.add_argument("--valid-out", default="data/valid.jsonl")
    args = parser.parse_args()

    rows, stats = load_rows(args.input, args.max_length)
    print(f"total rows read:      {stats['total']}")
    print(f"dropped (empty):      {stats['empty']}")
    print(f"dropped (duplicate):  {stats['duplicate']}")
    print(f"dropped (> {args.max_length} chars): {stats['too_long']}")
    print(f"kept:                 {stats['kept']}")

    train, valid = stratified_split(rows, args.valid_fraction, args.seed)

    def label_counts(subset):
        counts = {}
        for row in subset:
            counts[row["completion"]] = counts.get(row["completion"], 0) + 1
        return counts

    print(f"\ntrain: {len(train)} rows, class balance: {label_counts(train)}")
    print(f"valid: {len(valid)} rows, class balance: {label_counts(valid)}")

    Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.train_out, train)
    write_jsonl(args.valid_out, valid)
    print(f"\nwrote {args.train_out} and {args.valid_out}")


if __name__ == "__main__":
    main()
