from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import requests

BASE_URL = "https://apidatos.ree.es/en/datos/demanda/evolucion"
PARAMS = {
    "time_trunc": "hour",
    "geo_trunc": "electric_system",
    "geo_limit": "baleares",
    "geo_ids": 8742,
}

def fetch_range(start: date, end: date) -> list[dict]:
    """One API call. Raises requests.HTTPError if the API rejects the range."""
    response = requests.get(
        BASE_URL,
        params={
            **PARAMS,
            "start_date": start.strftime("%Y-%m-%dT00:00"),
            "end_date": end.strftime("%Y-%m-%dT00:00"),
        },
    )
    response.raise_for_status()
    return response.json()["included"][0]["attributes"]["values"]

def fetch_range_robust(start: date, end: date) -> list[dict]:
    """
    Fetch [start, end). If the API rejects the whole range, split it in
    half and retry each half. Stops splitting (reports a genuine
    failure) once a single day still fails.
    """
    try:
        return fetch_range(start, end)
    except requests.HTTPError:
        span_days = (end - start).days
        if span_days <= 1:
            print(f"  GENUINE FAILURE (single day, still fails): {start}")
            return []
        mid = start + timedelta(days=span_days // 2)
        print(f"  {start} -> {end} failed, splitting: {start}->{mid}, {mid}->{end}")
        return fetch_range_robust(start, mid) + fetch_range_robust(mid, end)


def save_chunk(readings: list[dict], csv_path: Path) -> None:
    """Append straight to disk. Called after every fetch, so a crash
    mid-run loses at most one chunk, never the whole backfill."""
    if not readings:
        return
    df_chunk = pd.DataFrame(readings)
    df_chunk["datetime"] = pd.to_datetime(df_chunk["datetime"], utc=True)
    df_chunk = df_chunk[["datetime", "value"]]
    file_exists = csv_path.exists()
    df_chunk.to_csv(csv_path, mode="a", header=not file_exists, index=False)
