"""
ACN-Data API Downloader (3-Month Date-Window Batches)
Avoids MongoDB/Eve deep-pagination timeouts on higher page numbers by splitting the
query into 3-month date intervals using the 'where' filter on connectionTime.

Usage:
  python download_acn_batched.py [--site caltech|jpl|office001] [--start-year 2018] [--end-year 2021] [--outdir data]
"""

import os
import json
import math
import argparse
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
import concurrent.futures
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def get_retry_session() -> requests.Session:
    """Configures a requests Session with exponential backoff retries."""
    session = requests.Session()
    retry_strategy = Retry(
        total=8,
        connect=8,
        read=8,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def format_http_date(dt: datetime) -> str:
    """Formats a datetime object to the RFC 1123 GMT string required by the ACN Eve API."""
    return dt.strftime("%a, %d %b %Y %H:%M:%S GMT")


def build_where_clause(start_dt: datetime, end_dt: datetime) -> str:
    """Builds the MongoDB/Eve 'where' filter string for a date window."""
    start_str = format_http_date(start_dt)
    end_str = format_http_date(end_dt)
    return f'connectionTime>="{start_str}" and connectionTime<"{end_str}"'


def fetch_batch_page(
    session: requests.Session,
    url: str,
    api_key: str,
    where_clause: str,
    page_num: int,
    max_results: int = 100
) -> list:
    """Fetches a single page within a 3-month batch."""
    params = {
        "where": where_clause,
        "page": page_num,
        "max_results": max_results,
        "sort": "connectionTime"
    }
    try:
        resp = session.get(url, auth=(api_key, ""), params=params, timeout=60)
        if resp.status_code == 200:
            return resp.json().get("_items", [])
        else:
            print(f"  [!] Page {page_num} returned HTTP {resp.status_code}")
    except Exception as e:
        print(f"  [!] Page {page_num} error: {e}")
    return []


def download_in_3month_batches(
    site: str,
    outdir: str,
    start_year: int = 2018,
    end_year: int = 2021,
    include_timeseries: bool = False,
    workers: int = 6
):
    api_key = os.environ.get("ACN_API_KEY")
    if not api_key:
        raise ValueError("ACN_API_KEY environment variable is not set.")

    os.makedirs(outdir, exist_ok=True)
    base_url = "https://ev.caltech.edu/api/v1/sessions"
    url = f"{base_url}/{site}/ts" if include_timeseries else f"{base_url}/{site}"

    session = get_retry_session()
    all_sessions_by_id = {}

    # Generate 3-month windows from start_year-01-01 to (end_year+1)-01-01
    current_start = datetime(start_year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    final_end = datetime(end_year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    batch_idx = 1
    while current_start < final_end:
        current_end = min(current_start + relativedelta(months=3), final_end)
        where_clause = build_where_clause(current_start, current_end)
        label = f"{current_start.strftime('%Y-%m-%d')} -> {current_end.strftime('%Y-%m-%d')}"

        # Probe page 1 of this 3-month batch
        probe_params = {
            "where": where_clause,
            "page": 1,
            "max_results": 100,
            "sort": "connectionTime"
        }
        try:
            resp = session.get(url, auth=(api_key, ""), params=probe_params, timeout=60)
        except Exception as e:
            print(f"[Batch {batch_idx} | {label}] Probe failed: {e}")
            current_start = current_end
            batch_idx += 1
            continue

        if resp.status_code != 200:
            print(f"[Batch {batch_idx} | {label}] HTTP {resp.status_code}: {resp.text}")
            current_start = current_end
            batch_idx += 1
            continue

        data = resp.json()
        meta = data.get("_meta", {})
        total_in_batch = meta.get("total", 0)
        actual_page_size = meta.get("max_results", 25)

        if total_in_batch == 0:
            print(f"[Batch {batch_idx:02d} | {label}] 0 sessions.")
            current_start = current_end
            batch_idx += 1
            continue

        total_pages = math.ceil(total_in_batch / actual_page_size)
        print(
            f"[Batch {batch_idx:02d} | {label}] Found {total_in_batch} sessions "
            f"({total_pages} pages @ {actual_page_size}/page)..."
        )

        batch_items = data.get("_items", [])
        remaining_pages = list(range(2, total_pages + 1))

        if remaining_pages:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        fetch_batch_page,
                        session,
                        url,
                        api_key,
                        where_clause,
                        p,
                        actual_page_size
                    ): p
                    for p in remaining_pages
                }
                for future in concurrent.futures.as_completed(futures):
                    items = future.result()
                    if items:
                        batch_items.extend(items)

        # Deduplicate by session _id and checkpoint to disk after each 3-month batch
        for item in batch_items:
            sid = item.get("_id", item.get("sessionID"))
            if sid:
                all_sessions_by_id[sid] = item

        print(
            f"  -> Batch complete: +{len(batch_items)} sessions "
            f"(Running unique total: {len(all_sessions_by_id)})"
        )

        # Save incremental checkpoint so progress is never lost
        checkpoint_file = os.path.join(
            outdir, f"acn_data_{site}_full{'_ts' if include_timeseries else ''}.json"
        )
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "_meta": {"total": len(all_sessions_by_id)},
                    "_items": list(all_sessions_by_id.values())
                },
                f,
                indent=2
            )

        current_start = current_end
        batch_idx += 1

    final_filepath = os.path.join(
        outdir, f"acn_data_{site}_full{'_ts' if include_timeseries else ''}.json"
    )
    print(f"\nAll 3-month batches finished! Total unique sessions saved: {len(all_sessions_by_id)}")
    print(f"Output file: {final_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download ACN-Data in 3-month batches to avoid deep-pagination timeouts."
    )
    parser.add_argument(
        "--site",
        type=str,
        choices=["caltech", "jpl", "office001"],
        default="caltech",
        help="The ACN site to download data for (default: caltech)."
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2018,
        help="Start year for 3-month batches (default: 2018, when ACN-Data collection began)."
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2021,
        help="End year (inclusive) for 3-month batches (default: 2021)."
    )
    parser.add_argument(
        "--timeseries",
        action="store_true",
        help="Include high-resolution charging current time series data."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data",
        help="Directory to save the downloaded JSON file (default: 'data')."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Concurrent worker threads per 3-month batch (default: 6)."
    )

    args = parser.parse_args()
    download_in_3month_batches(
        site=args.site,
        outdir=args.outdir,
        start_year=args.start_year,
        end_year=args.end_year,
        include_timeseries=args.timeseries,
        workers=args.workers
    )