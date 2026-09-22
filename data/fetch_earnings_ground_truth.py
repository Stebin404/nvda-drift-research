"""
data/fetch_earnings_ground_truth.py

ONE-TIME provenance script. Run this locally to build a per-ticker
earnings-dates CSV from a company's official SEC EDGAR filing history.

Why SEC EDGAR and not a finance aggregator site:
  - Public companies are legally required to file an 8-K with
    Item 2.02 ("Results of Operations and Financial Condition") within
    days of each earnings release. The 8-K filing date is a matter of
    public record, not a derived/estimated statistic.
  - This makes the ground truth genuinely external to the modeling
    pipeline: it does not depend on KS tests, PSI, volatility, or any
    other statistic computed elsewhere in this codebase.

What this script does:
  1. Pulls the full filing index for the given ticker/CIK from SEC's
     submissions API.
  2. Filters to Form 8-K filings.
  3. Writes a CSV with columns: date, source_url, filing_type.

IMPORTANT: an 8-K is filed for many reasons (earnings, exec changes,
M&A, restatements, etc.), not only earnings releases. This script does
NOT auto-classify which 8-Ks are earnings announcements -- that
requires a manual cross-check against the company's investor relations
press release archive, the same as was done for NVDA
(data/classify_earnings_candidates.py / data/finalize_earnings_dates.py).
Add an `is_earnings` column to the output CSV and filter to True rows
before treating this as final ground truth.

Usage:
    python data/fetch_earnings_ground_truth.py --ticker NVDA --cik 0001045810
    python data/fetch_earnings_ground_truth.py --ticker AMD --cik 0000002488
    python data/fetch_earnings_ground_truth.py --ticker TSLA --cik 0001318605
    python data/fetch_earnings_ground_truth.py --ticker JNJ --cik 0000200406

CIKs used in this project (verified against SEC EDGAR CIK lookup):
    NVDA  0001045810
    AMD   0000002488
    TSLA  0001318605
    JNJ   0000200406

Requires internet access to data.sec.gov and the requests package.
SEC requires a descriptive User-Agent header identifying your
application/contact -- replace the placeholder below with your real
contact info before running, or SEC may rate-limit / block the
request.
"""

import argparse
import csv
import time
import requests

# SEC requires a descriptive User-Agent with contact info for programmatic
# access. Replace this before running -- generic/missing UAs get throttled.
HEADERS = {
    "User-Agent": "stock-drift-monitor research project (stebinlimson@gmail.com)"
}


def fetch_submissions(cik):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def extract_8k_filings(submissions_json, cik):
    """
    The 'recent' block covers the latest ~1000 filings. Older filings
    live in separate 'files' referenced under submissions_json
    ['filings']['files']. We walk both to get full historical coverage.
    """
    all_filings = []

    recent = submissions_json["filings"]["recent"]
    all_filings.append(recent)

    for older_file in submissions_json["filings"].get("files", []):
        older_url = f"https://data.sec.gov/submissions/{older_file['name']}"
        resp = requests.get(older_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        all_filings.append(resp.json())
        time.sleep(0.2)  # be polite to SEC's rate limits

    eight_ks = []
    for block in all_filings:
        forms = block["form"]
        dates = block["filingDate"]
        accession_numbers = block["accessionNumber"]
        primary_docs = block["primaryDocument"]

        for form, date, accession, primary_doc in zip(
            forms, dates, accession_numbers, primary_docs
        ):
            if form == "8-K":
                accession_nodash = accession.replace("-", "")
                url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{int(cik)}/{accession_nodash}/{primary_doc}"
                )
                eight_ks.append({
                    "date": date,
                    "source_url": url,
                    "filing_type": "8-K",
                })

    return eight_ks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="e.g. NVDA, AMD, TSLA, JNJ")
    parser.add_argument("--cik", required=True, help="10-digit SEC CIK, e.g. 0000002488")
    args = parser.parse_args()

    output_path = f"data/earnings_dates_{args.ticker.lower()}.csv"

    submissions = fetch_submissions(args.cik)
    eight_ks = extract_8k_filings(submissions, args.cik)

    eight_ks.sort(key=lambda r: r["date"])

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "source_url", "filing_type"])
        writer.writeheader()
        writer.writerows(eight_ks)

    print(f"Wrote {len(eight_ks)} 8-K filing dates for {args.ticker} to {output_path}")
    print(
        "\nIMPORTANT: An 8-K is filed for many reasons (earnings, exec "
        "changes, M&A, etc.), not only earnings. Open the CSV and "
        "manually flag which rows are quarterly earnings releases "
        f"(cross-reference against {args.ticker}'s investor relations "
        "press release archive) before treating this as final ground "
        "truth -- same process as data/classify_earnings_candidates.py "
        "/ data/finalize_earnings_dates.py did for NVDA."
    )


if __name__ == "__main__":
    main()