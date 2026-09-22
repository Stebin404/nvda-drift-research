"""
evaluation/ground_truth_external.py

Replaces a circular ground-truth approach (deriving "regime change"
labels from a statistical test on the same data being evaluated) with
labels sourced from NVIDIA's actual quarterly earnings announcement
dates, individually verified against SEC EDGAR 8-K filings.

See data/earnings_dates_FINAL.csv for the final, manually-reviewed
ground truth (32 confirmed earnings dates, 2018-2026), and
data/fetch_earnings_ground_truth.py / data/classify_earnings_candidates.py
/ data/finalize_earnings_dates.py for the full provenance pipeline that
produced it.

Why earnings dates are a defensible external ground truth:
  1. They are scheduled, publicly verifiable, and occur independent of
     price action -- the date is not derived from price data.
  2. Post-earnings-announcement drift is a documented phenomenon in the
     finance literature.
  3. They are NOT derived from any statistic (KS, PSI, ADWIN, PH,
     KSWIN) used elsewhere in this pipeline, so there is no circularity.

LIMITATION TO STATE EXPLICITLY IN THE PAPER: earnings dates are a
proxy for regime change, not a perfect ground truth. A detector could
legitimately fail to fire near an earnings date (market had already
priced in the result) or fire near a date with no earnings event (a
macro shock -- e.g. the August 26, 2022 export-control disclosure
found during manual review, which was correctly excluded from this
ground truth but is itself a real, externally-verifiable shock worth
discussing as a secondary case study).
"""

import pandas as pd

DEFAULT_GROUND_TRUTH_PATH = "data/earnings_dates_FINAL.csv"


def load_earnings_dates(csv_path=DEFAULT_GROUND_TRUTH_PATH):
    """
    Loads the final, confirmed earnings-date CSV. Expects a 'date'
    column (YYYY-MM-DD). Every row in this file has already been
    manually verified as a genuine earnings release -- there is no
    is_earnings filtering needed here, unlike the intermediate
    candidates file.

    Raises a clear error rather than silently returning an empty
    result if the file doesn't exist -- a silently-empty ground truth
    would make recall trivially undefined and is a dangerous failure
    mode to have go unnoticed.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find ground-truth CSV at '{csv_path}'. Run "
            f"data/fetch_earnings_ground_truth.py, "
            f"data/classify_earnings_candidates.py, and "
            f"data/finalize_earnings_dates.py first."
        ) from e

    df["date"] = pd.to_datetime(df["date"])
    return df


def map_dates_to_row_indices(price_df, earnings_dates_df, date_column="Date"):
    """
    Converts earnings calendar dates into row indices within price_df.

    Earnings dates won't always land exactly on a trading day. For each
    earnings date, we map to the nearest trading day AT OR AFTER the
    earnings date -- matching the convention that after-market earnings
    releases move price action starting the next session.

    Returns a sorted, deduplicated list of integer row positions
    (relative to price_df's own row ordering), directly comparable to
    the integer positions produced by the streaming detectors.
    """
    price_dates = pd.to_datetime(price_df[date_column]).reset_index(drop=True)

    row_indices = []
    unmatched = []

    for earnings_date in earnings_dates_df["date"]:
        candidates = price_dates[price_dates >= earnings_date]

        if len(candidates) == 0:
            unmatched.append(earnings_date)
            continue

        nearest_idx = candidates.index[0]
        row_indices.append(int(nearest_idx))

    if unmatched:
        print(
            f"WARNING: {len(unmatched)} earnings date(s) fell after the "
            f"end of the available price history and were dropped: "
            f"{unmatched}"
        )

    return sorted(set(row_indices))


def get_earnings_ground_truth(price_df, csv_path=DEFAULT_GROUND_TRUTH_PATH, date_column="Date"):
    """
    Convenience entry point: load earnings dates and map them to row
    indices in price_df in one call.

    NOTE: positions returned are relative to price_df's own row
    ordering. If price_df is the FULL dataframe, positions are
    full-dataframe-relative. Detector alarms are always test-fold-
    relative (detectors only ever observe a sliced-out test stream),
    so for evaluating detectors, use
    get_earnings_ground_truth_for_test_fold below instead.
    """
    earnings_df = load_earnings_dates(csv_path)
    return map_dates_to_row_indices(price_df, earnings_df, date_column=date_column)


def get_earnings_ground_truth_for_test_fold(
    full_df,
    test_fold_start_index,
    test_fold_end_index,
    csv_path=DEFAULT_GROUND_TRUTH_PATH,
    date_column="Date",
):
    """
    Returns earnings-derived ground truth positions in the SAME
    coordinate system as detector alarms: integer positions relative
    to the start of the test fold, not the full dataframe.

    THE COORDINATE MISMATCH THIS SOLVES: detectors only ever see a
    sliced-out test-fold stream. An alarm at position 50 means "50
    rows into the test fold," not "row 50 of the full dataframe."
    Earnings dates, by contrast, map naturally onto full-dataframe
    positions. This function performs that coordinate shift once,
    explicitly, at the evaluation boundary.

    Parameters
    ----------
    full_df : pd.DataFrame
        The full dataframe (post create_features / create_target),
        BEFORE any train/test split or fold slicing.
    test_fold_start_index : int
        The row index in full_df where the test fold begins. Must
        match exactly the same boundary used to slice the stream
        handed to the detectors for this fold.

    Returns
    -------
    list[int]
        Sorted, deduplicated positions relative to test_fold_start_index.
        Earnings dates falling in the TRAINING portion of full_df are
        dropped (not observable to any detector) with a warning.
    """
    earnings_df = load_earnings_dates(csv_path)

    full_indices = map_dates_to_row_indices(full_df, earnings_df, date_column=date_column)

    in_test_fold = [
        idx for idx in full_indices
        if test_fold_start_index <= idx < test_fold_end_index
    ]
    dropped = len(full_indices) - len(in_test_fold)

    if dropped > 0:
        print(
            f"WARNING: {dropped} earnings date(s) fall outside this "
            f"fold's test window (rows {test_fold_start_index} to "
            f"{test_fold_end_index}) and are not observable to this "
            f"fold's detectors. Excluded from this fold's ground truth."
        )

    test_fold_relative = sorted(set(idx - test_fold_start_index for idx in in_test_fold))
    return test_fold_relative