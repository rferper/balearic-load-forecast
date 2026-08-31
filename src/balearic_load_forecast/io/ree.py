"""Client for the REE public demand API."""

from datetime import date, timedelta

import requests
from loguru import logger

BASE_URL = "https://apidatos.ree.es/en/datos/demanda/evolucion"
TIMEOUT_SECONDS = 60

GEO_PARAMS = {
    "time_trunc": "hour",
    "geo_trunc": "electric_system",
    "geo_limit": "baleares",
    "geo_ids": 8742,
}


def fetch_range(start: date, end: date) -> list[dict]:
    """Fetch hourly readings in `[start, end)` with one API call."""
    response = requests.get(
        BASE_URL,
        params={
            **GEO_PARAMS,
            "start_date": start.strftime("%Y-%m-%dT00:00"),
            "end_date": end.strftime("%Y-%m-%dT00:00"),
        },
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["included"][0]["attributes"]["values"]


def fetch_range_robust(start: date, end: date) -> list[dict]:
    """Fetch `[start, end)`, halving the range when the API rejects it.

    Wide hourly ranges are refused with a 502, so the range is bisected until
    accepted. A single day that still fails is a gap in the source: it is
    logged and skipped rather than aborting the run.
    """
    try:
        return fetch_range(start, end)
    except requests.HTTPError:
        span_days = (end - start).days
        if span_days <= 1:
            logger.warning("REE has no data for {}, skipping", start)
            return []
        mid = start + timedelta(days=span_days // 2)
        logger.debug("{} -> {} refused, splitting at {}", start, end, mid)
        return fetch_range_robust(start, mid) + fetch_range_robust(mid, end)
