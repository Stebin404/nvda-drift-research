"""
data/classify_earnings_candidates.py

Filters the full 8-K list down to a small set of CANDIDATE earnings
dates, based on NVIDIA's known quarterly reporting pattern, so manual
verification only needs to cover a short list instead of the full
filing history.

FIX: bounded to 2018-01-01 onwards, matching the actual research
study period (data/loader.py pulls NVDA data from 2018 onwards). The
original version of this script had no date floor, so it pulled in
candidate rows back to 2002 -- 16+ years of filings completely outside
the study period, and some of those very old filings also hit broken/
reorganized links in SEC's archive (the "NoSuchKey" error), which is
unrelated to anything in this project and not worth debugging since
those rows shouldn't be in scope at all.

NVIDIA's fiscal year runs Feb-Jan (offset from the calendar year), so
its quarterly earnings releases historically cluster in four narrow
windows each year:
  Q4/FY-end results : mid-to-late February
  Q1 results         : mid-to-late May
  Q2 results         : mid-to-late August
  Q3 results         : mid-to-late November

This script does NOT claim these windows are exhaustive or perfectly
precise -- exact dates shift by a few days year to year. It uses a
generous +/- window around each typical period to avoid missing a
real earnings date, which means some false positives (non-earnings
8-Ks that happen to fall in these windows) may still appear in the
output. That's fine -- this script's job is to shrink the full list
to a short list for YOU to manually confirm via source_url, not to
make the final call automatically.

Output: data/earnings_candidates_FOR_REVIEW.csv
    A short list of candidate rows with an empty 'is_earnings' column
    for you to fill in (TRUE/FALSE) after checking each source_url.
"""

import pandas as pd

INPUT_PATH = "data/earnings_dates.csv"
OUTPUT_PATH = "data/earnings_candidates_FOR_REVIEW.csv"

STUDY_PERIOD_START = pd.Timestamp("2018-01-01")

CANDIDATE_WINDOWS = [
    (2, 10, 28),   # mid-to-late February
    (5, 15, 31),   # mid-to-late May
    (8, 15, 31),   # mid-to-late August
    (11, 10, 30),  # mid-to-late November
]


def is_candidate_date(date):
    for month, day_start, day_end in CANDIDATE_WINDOWS:
        if date.month == month and day_start <= date.day <= day_end:
            return True
    return False


def main():
    df = pd.read_csv(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"])

    in_study_period = df[df["date"] >= STUDY_PERIOD_START].copy()
    excluded_count = len(df) - len(in_study_period)

    candidates = in_study_period[in_study_period["date"].apply(is_candidate_date)].copy()
    candidates = candidates.sort_values("date").reset_index(drop=True)

    candidates["is_earnings"] = ""

    candidates.to_csv(OUTPUT_PATH, index=False)

    print(f"Total 8-K filings in source file: {len(df)}")
    print(f"Excluded (before study period start {STUDY_PERIOD_START.date()}): {excluded_count}")
    print(f"Candidates within study period matching quarterly windows: {len(candidates)}")
    print(f"Candidates written to {OUTPUT_PATH}")
    print(
        "\nNext: open this file, open each source_url in a browser, "
        "confirm whether it's a quarterly earnings release (mentions "
        "'Results of Operations' / quarterly revenue), and fill in "
        "TRUE or FALSE in the is_earnings column for every row."
    )
    print(
        "\nSanity check: 2018 onwards is about 8 years, so expect "
        "somewhere around 28-32 TRUE rows. If far outside that range, "
        "double check for a missing quarter or a false positive."
    )


if __name__ == "__main__":
    main()