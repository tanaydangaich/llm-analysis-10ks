"""
Download 10-K and 10-Q filings from SEC EDGAR for a given ticker or CIK.
"""
import argparse
from pathlib import Path
from sec_edgar_downloader import Downloader


def fetch(ticker: str, filing_types: list[str], num_filings: int, out_dir: Path) -> None:
    dl = Downloader("MyCompany", "admin@example.com", out_dir)
    for ftype in filing_types:
        print(f"Fetching {num_filings}x {ftype} for {ticker}...")
        dl.get(ftype, ticker, limit=num_filings)
        print(f"  -> saved to {out_dir / 'sec-edgar-filings' / ticker / ftype}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SEC filings")
    parser.add_argument("--ticker", default="DCTK", help="Stock ticker symbol")
    parser.add_argument("--types", nargs="+", default=["10-K", "10-Q"], help="Filing types")
    parser.add_argument("--limit", type=int, default=3, help="Number of filings per type")
    parser.add_argument("--out", default="data/raw", help="Output directory")
    args = parser.parse_args()

    fetch(
        ticker=args.ticker,
        filing_types=args.types,
        num_filings=args.limit,
        out_dir=Path(args.out),
    )
