"""
data/finalize_earnings_dates.py

Merges the manually-reviewed earnings candidates
(earnings_candidates_FOR_REVIEW.csv, with is_earnings filled in as
TRUE/FALSE) into a final, clean ground-truth file:
data/earnings_dates_FINAL.csv

This file contains ONLY confirmed quarterly earnings dates -- every
row has been individually verified against the actual SEC 8-K filing
content (Item 2.02, "Results of Operations"), not inferred from
filename pattern or date alone. See the candidates file's source_url
column for the original filing if this needs to be re-verified later.

Output: data/earnings_dates_FINAL.csv
    Columns: date, source_url
    (filing_type and is_earnings dropped, since every remaining row
    is_earnings == TRUE by construction)

This is deliberately written to a NEW file (earnings_dates_FINAL.csv)
rather than overwriting earnings_dates.csv directly, so the original
237-row full 8-K list is preserved for any future re-classification
or audit, and the final file is named distinctly to avoid confusion
between "all 8-Ks" and "confirmed earnings only."

After running this, evaluation/ground_truth_external.py should point
at earnings_dates_FINAL.csv as the canonical ground-truth source.
"""

import pandas as pd

CANDIDATES_PATH = "data/earnings_candidates_FOR_REVIEW.csv"
OUTPUT_PATH = "data/earnings_dates_FINAL.csv"


def main():
    df = pd.read_csv(CANDIDATES_PATH)

    # Normalize the is_earnings column to handle any case variation
    # (TRUE, true, True) or stray whitespace from manual editing.
    df["is_earnings"] = df["is_earnings"].astype(str).str.strip().str.upper()

    unexpected_values = set(df["is_earnings"]) - {"TRUE", "FALSE"}
    if unexpected_values:
        raise ValueError(
            f"Found unexpected is_earnings values: {unexpected_values}. "
            f"Every row must be exactly TRUE or FALSE. Fix "
            f"{CANDIDATES_PATH} before re-running this script."
        )

    confirmed = df[df["is_earnings"] == "TRUE"].copy()
    rejected_count = len(df) - len(confirmed)

    confirmed = confirmed[["date", "source_url"]].sort_values("date").reset_index(drop=True)

    confirmed.to_csv(OUTPUT_PATH, index=False)

    print(f"Total reviewed candidates: {len(df)}")
    print(f"Confirmed earnings dates (TRUE): {len(confirmed)}")
    print(f"Rejected as non-earnings (FALSE): {rejected_count}")
    print(f"\nFinal ground-truth file written to: {OUTPUT_PATH}")
    print(
        "\nNext step: update evaluation/ground_truth_external.py's "
        "default csv_path to point at this file instead of "
        "data/earnings_dates.csv, or pass csv_path explicitly when "
        "calling its functions."
    )


if __name__ == "__main__":
    main()