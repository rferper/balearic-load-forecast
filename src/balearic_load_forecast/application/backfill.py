"""Download the historical demand series from REE.

Each chunk is appended as it arrives and a re-run resumes from the newest day
already stored, so an interrupted run costs one chunk rather than the lot.
"""

import time
from datetime import date, timedelta

from loguru import logger

from ..io import datasets, ree
from ..settings import BackfillConfig


def run(config: BackfillConfig) -> int:
    """Fetch every missing day up to today and return the rows appended."""
    last_day = datasets.last_demand_day(config.raw_csv)
    current = last_day + timedelta(days=1) if last_day else config.start
    # Local calendar day: REE publishes against Spanish local dates.
    final_end = date.today()

    if current >= final_end:
        logger.info("already current through {}, nothing to fetch", last_day)
        return 0

    logger.info("fetching {} -> {}", current, final_end)
    written = 0
    while current < final_end:
        chunk_end = min(current + timedelta(days=config.chunk_days), final_end)
        readings = ree.fetch_range_robust(current, chunk_end)
        written += datasets.append_readings(readings, config.raw_csv)
        current = chunk_end
        time.sleep(config.pause_seconds)

    logger.success("appended {} rows to {}", written, config.raw_csv)
    return written
