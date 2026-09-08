"""
data/fetch_earnings_ground_truth.py

ONE-TIME provenance script. Run this locally to build
data/earnings_dates.csv from NVIDIA's official SEC EDGAR filing
history.

Why SEC EDGAR and not a finance aggregator site:
  - NVIDIA (CIK 0001045810) is legally required to file an 8-K with
    Item 2.02 ("Results of Operations and Financial Condition") within
    days of each earnings release. The 8-K filing date is a matter of
    public record, not a derived/estimated statistic.
  - This makes the ground truth genuinely external to the modeling
    pipeline: it does not depend on KS tests, PSI, volatility, or any
    other statistic computed elsewhere in this codebase.

What this script does:
  1. Pulls the full filing index for NVDA from SEC's submissions API.
  2. Filters to Form 8-K filings.
  3. Writes a CSV with columns: date, source_url, filing_type.

IMPORTANT: an 8-K is filed for many reasons (earnings, exec changes,
M&A, restatements, etc.), not only earnings releases. This script does
NOT auto-classify which 8-Ks are earnings announcements -- that
requires a manual cross-check against NVIDIA's investor relations
press release archive (investor.nvidia.com). Add an `is_earnings`
column to the output CSV and filter to True rows before treating this
as final ground truth.

Usage:
    python data/fetch_earnings_ground_truth.py

Requires internet access to data.sec.gov and the requests package.
SEC requires a descriptive User-Agent header identifying your
application/contact -- replace the placeholder below with your real
contact info before running, or SEC may rate-limit / block the
request.
"""

import csv
import time
import requests

NVDA_CIK = "0001045810"  # NVIDIA Corporation, verified filer ID on EDGAR

# SEC requires a descriptive User-Agent with contact info for programmatic
# access. Replace this before running -- generic/missing UAs get throttled.
HEADERS = {
    "User-Agent": "stock-drift-monitor research project (stebinlimson@gmail.com)"
}

SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{NVDA_CIK}.json"

OUTPUT_PATH = "data/earnings_dates.csv"


def fetch_submissions():
    resp = requests.get(SUBMISSIONS_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def extract_8k_filings(submissions_json):
    """
    The 'recent' block covers the latest ~1000 filings. Older filings
    live in separate 'files' referenced under submissions_json
    ['filings']['files']. We walk both to get full 2018-onwards
    coverage.
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
                    f"{int(NVDA_CIK)}/{accession_nodash}/{primary_doc}"
                )
                eight_ks.append({
                    "date": date,
                    "source_url": url,
                    "filing_type": "8-K",
                })

    return eight_ks


def main():
    submissions = fetch_submissions()
    eight_ks = extract_8k_filings(submissions)

    eight_ks.sort(key=lambda r: r["date"])

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "source_url", "filing_type"])
        writer.writeheader()
        writer.writerows(eight_ks)

    print(f"Wrote {len(eight_ks)} 8-K filing dates to {OUTPUT_PATH}")
    print(
        "\nIMPORTANT: An 8-K is filed for many reasons (earnings, exec "
        "changes, M&A, etc.), not only earnings. Open the CSV and "
        "manually flag which rows are quarterly earnings releases "
        "(cross-reference against NVIDIA's investor relations press "
        "release archive at investor.nvidia.com) before treating this "
        "as your final ground truth."
    )


if __name__ == "__main__":
    main()