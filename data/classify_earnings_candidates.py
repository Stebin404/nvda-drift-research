"""
data/classify_earnings_candidates.py

Filters the full 8-K list down to a small set of CANDIDATE earnings
dates, based on a company's known quarterly reporting pattern, so
manual verification only needs to cover a short list instead of the
full filing history.

FIX: bounded to 2018-01-01 onwards, matching the actual research
study period (data/loader.py pulls price data from 2018 onwards). The
original version of this script had no date floor, so it pulled in
candidate rows back to 2002 -- 16+ years of filings completely outside
the study period, and some of those very old filings also hit broken/
reorganized links in SEC's archive (the "NoSuchKey" error), which is
unrelated to anything in this project and not worth debugging since
those rows shouldn't be in scope at all.

NVIDIA's fiscal year runs Feb-Jan (offset from the calendar year), so
its quarterly earnings releases historically cluster in four narrow
windows each year (mid-to-late Feb/May/Aug/Nov). AMD, Tesla, and JNJ
all run standard calendar fiscal years and report roughly 3-5 weeks
after each quarter-end (late Jan-early Feb, late Apr-early May, late
Jul-early Aug, late Oct-early Nov) -- confirmed for AMD directly
against its own historical announcement-date table (2018-2025); TSLA
and JNJ use the same generic calendar-quarter window as a starting
point, NOT independently confirmed the same way AMD was. Treat the
TSLA/JNJ window as a wider net to manually verify against source_url,
same as this script already does for NVDA -- do not skip the manual
check just because a ticker isn't NVDA.

This script does NOT claim these windows are exhaustive or perfectly
precise -- exact dates shift by a few days year to year. It uses a
generous +/- window around each typical period to avoid missing a
real earnings date, which means some false positives (non-earnings
8-Ks that happen to fall in these windows) may still appear in the
output. That's fine -- this script's job is to shrink the full list
to a short list for YOU to manually confirm via source_url, not to
make the final call automatically.

Usage:
    python data/classify_earnings_candidates.py --ticker NVDA
    python data/classify_earnings_candidates.py --ticker AMD
    python data/classify_earnings_candidates.py --ticker TSLA
    python data/classify_earnings_candidates.py --ticker JNJ

Input:  data/earnings_dates_<ticker>.csv (NVDA's original file is
        data/earnings_dates.csv, kept as-is for backward compatibility)
Output: data/earnings_candidates_FOR_REVIEW_<ticker>.csv
    A short list of candidate rows with an empty 'is_earnings' column
    for you to fill in (TRUE/FALSE) after checking each source_url.
"""

import argparse
import pandas as pd

STUDY_PERIOD_START = pd.Timestamp("2018-01-01")

# Per-ticker candidate windows: (month, day_start, day_end).
# NVDA confirmed against its own fiscal-year offset (see repo history).
# AMD confirmed directly against AMD's own reported announcement dates
# 2018-2025 (alphaquery.com historical table, cross-checked against
# SEC 8-K filing dates). TSLA/JNJ are a generic calendar-quarter
# starting window, NOT independently confirmed per-ticker -- widen
# these if the manual review turns up dates falling just outside them.
CANDIDATE_WINDOWS = {
    "NVDA": [
        (2, 10, 28),   # mid-to-late February
        (5, 15, 31),   # mid-to-late May
        (8, 15, 31),   # mid-to-late August
        (11, 10, 30),  # mid-to-late November
    ],
    "AMD": [
        (1, 20, 31),   # late January
        (2, 1, 10),    # early February (spillover from late Jan)
        (4, 20, 30),   # late April
        (5, 1, 10),    # early May
        (7, 20, 31),   # late July
        (8, 1, 10),    # early August
        (10, 20, 31),  # late October
        (11, 1, 10),   # early November
    ],
    "TSLA": [
        (1, 15, 31),
        (2, 1, 10),   # early Feb spillover (e.g. 2018-02-07)
        (4, 15, 30),
        (5, 1, 10),   # early May spillover (e.g. 2018-05-02)
        (7, 15, 31),
        (8, 1, 10),   # early Aug spillover (e.g. 2018-08-01)
        (10, 15, 31),
    ],
    "JNJ": [
        (1, 10, 25),
        (4, 10, 25),
        (7, 10, 25),
        (10, 10, 25),
    ],
}


def is_candidate_date(date, windows):
    for month, day_start, day_end in windows:
        if date.month == month and day_start <= date.day <= day_end:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, choices=list(CANDIDATE_WINDOWS))
    args = parser.parse_args()
    ticker = args.ticker

    input_path = "data/earnings_dates.csv" if ticker == "NVDA" else f"data/earnings_dates_{ticker.lower()}.csv"
    output_path = "data/earnings_candidates_FOR_REVIEW.csv" if ticker == "NVDA" else f"data/earnings_candidates_FOR_REVIEW_{ticker.lower()}.csv"
    windows = CANDIDATE_WINDOWS[ticker]

    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])

    in_study_period = df[df["date"] >= STUDY_PERIOD_START].copy()
    excluded_count = len(df) - len(in_study_period)

    candidates = in_study_period[in_study_period["date"].apply(lambda d: is_candidate_date(d, windows))].copy()
    candidates = candidates.sort_values("date").reset_index(drop=True)

    candidates["is_earnings"] = ""

    candidates.to_csv(output_path, index=False)

    print(f"Ticker: {ticker}")
    print(f"Total 8-K filings in source file: {len(df)}")
    print(f"Excluded (before study period start {STUDY_PERIOD_START.date()}): {excluded_count}")
    print(f"Candidates within study period matching quarterly windows: {len(candidates)}")
    print(f"Candidates written to {output_path}")
    print(
        "\nNext: open this file, open each source_url in a browser, "
        "confirm whether it's a quarterly earnings release (mentions "
        "'Results of Operations' / quarterly revenue), and fill in "
        "TRUE or FALSE in the is_earnings column for every row."
    )
    print(
        "\nSanity check: 2018 onwards is about 8 years, so expect "
        "somewhere around 28-32 TRUE rows. If far outside that range, "
        "double check for a missing quarter or a false positive -- "
        "especially for TSLA/JNJ, whose windows above are unconfirmed "
        "defaults, not independently verified the way NVDA/AMD's are."
    )


if __name__ == "__main__":
    main()