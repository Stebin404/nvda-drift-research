"""
data/generate_manifest.py

TODO item #6c (repo hygiene): both reviewers asked for the exact fold
boundaries and the SEC accession numbers behind the earnings-event
anchors to be inspectable directly, not only reproducible by running
the pipeline. This script writes two small, checked-in CSVs:

  data/fold_manifest.csv
      One row per (ticker, fold_id): the exact row-index and calendar-
      date boundaries of every rolling-origin fold used in
      evaluation/run_rolling_evaluation.py and
      evaluation/bridge_experiment.py (n_folds=5, min_train_fraction=
      0.5, test_fraction=0.08 -- the defaults used throughout the
      paper), plus the retraining-comparison windows from
      rebuild_retraining_tables.py (single 400-row window, and the
      three independent 300-row windows).

  data/earnings_accession_manifest.csv
      One row per verified earnings-event anchor (all three tickers):
      the date, the SEC accession number (parsed out of the 8-K
      source_url already recorded in
      data/earnings_dates_FINAL[_amd|_tsla].csv), and the source URL
      itself.

Run from the repo root: python data/generate_manifest.py
"""

import re

import pandas as pd

from data.loader import load_stock_data
from features.engineer import create_features
from features.target import create_target
from evaluation.rolling_origin import generate_rolling_folds

TICKERS = {
    "NVDA": "data/earnings_dates_FINAL.csv",
    "AMD": "data/earnings_dates_FINAL_amd.csv",
    "TSLA": "data/earnings_dates_FINAL_tsla.csv",
}

N_FOLDS = 5
MIN_TRAIN_FRACTION = 0.5
TEST_FRACTION = 0.08

ACCESSION_RE = re.compile(r"/data/\d+/(\d{18})/")


def build_fold_manifest():
    rows = []
    for ticker in TICKERS:
        df = load_stock_data(ticker)
        df = create_features(df)
        df = create_target(df)
        df = df.reset_index(drop=True)

        folds = generate_rolling_folds(
            len(df), n_folds=N_FOLDS,
            min_train_fraction=MIN_TRAIN_FRACTION, test_fraction=TEST_FRACTION,
        )

        for fold in folds:
            train_end_date = df["Date"].iloc[fold["train_end"] - 1]
            test_start_date = df["Date"].iloc[fold["test_start"]]
            test_end_date = df["Date"].iloc[min(fold["test_end"], len(df)) - 1]
            rows.append({
                "ticker": ticker,
                "experiment": "rolling_origin_tournament_and_bridge",
                "fold_id": fold["fold_id"],
                "train_end_row": fold["train_end"],
                "test_start_row": fold["test_start"],
                "test_end_row": fold["test_end"],
                "train_end_date": train_end_date.date().isoformat(),
                "test_start_date": test_start_date.date().isoformat(),
                "test_end_date": test_end_date.date().isoformat(),
                "n_rows_total": len(df),
            })

        # Retraining-comparison windows (rebuild_retraining_tables.py):
        # single 400-row window at the end of history, and three
        # independent, non-overlapping 300-row windows.
        eval_window = 400
        initial_train_size = len(df) - eval_window
        rows.append({
            "ticker": ticker,
            "experiment": "retraining_single_window",
            "fold_id": 0,
            "train_end_row": initial_train_size,
            "test_start_row": initial_train_size,
            "test_end_row": len(df),
            "train_end_date": df["Date"].iloc[initial_train_size - 1].date().isoformat(),
            "test_start_date": df["Date"].iloc[initial_train_size].date().isoformat(),
            "test_end_date": df["Date"].iloc[len(df) - 1].date().isoformat(),
            "n_rows_total": len(df),
        })

        window_size = 300
        n_windows = 3
        # Non-overlapping windows drawn from different phases of history,
        # spaced across the series (mirrors rebuild_retraining_tables.py).
        usable = len(df) - window_size
        starts = [int(usable * frac) for frac in (0.35, 0.60, 0.85)]
        for w_id, start in enumerate(starts):
            end = start + window_size
            rows.append({
                "ticker": ticker,
                "experiment": "retraining_multi_window",
                "fold_id": w_id,
                "train_end_row": start,
                "test_start_row": start,
                "test_end_row": end,
                "train_end_date": df["Date"].iloc[start - 1].date().isoformat(),
                "test_start_date": df["Date"].iloc[start].date().isoformat(),
                "test_end_date": df["Date"].iloc[end - 1].date().isoformat(),
                "n_rows_total": len(df),
            })

    return pd.DataFrame(rows)


def build_accession_manifest():
    rows = []
    for ticker, csv_path in TICKERS.items():
        dates_df = pd.read_csv(csv_path)
        for _, r in dates_df.iterrows():
            match = ACCESSION_RE.search(r["source_url"])
            accession = match.group(1) if match else None
            accession_formatted = (
                f"{accession[:10]}-{accession[10:12]}-{accession[12:]}" if accession else None
            )
            rows.append({
                "ticker": ticker,
                "date": r["date"],
                "sec_accession_number": accession_formatted,
                "source_url": r["source_url"],
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    fold_manifest = build_fold_manifest()
    fold_manifest.to_csv("data/fold_manifest.csv", index=False)
    print(f"Wrote data/fold_manifest.csv ({len(fold_manifest)} rows)")

    accession_manifest = build_accession_manifest()
    accession_manifest.to_csv("data/earnings_accession_manifest.csv", index=False)
    print(f"Wrote data/earnings_accession_manifest.csv ({len(accession_manifest)} rows)")

    n_missing = accession_manifest["sec_accession_number"].isna().sum()
    if n_missing:
        print(f"WARNING: {n_missing} row(s) had a source_url the accession regex could not parse.")
