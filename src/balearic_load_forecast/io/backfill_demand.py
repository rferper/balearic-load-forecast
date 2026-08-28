import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from balearic_load_forecast.io import readdata

CSV_PATH = Path("data/balearic_demand_raw.csv")


def backfill() -> None:
    Path("data").mkdir(exist_ok=True)

    if CSV_PATH.exists():
        existing = pd.read_csv(CSV_PATH)
        existing["datetime"] = pd.to_datetime(existing["datetime"], utc=True)
        current = existing["datetime"].max().date() + timedelta(days=1)
        print(f"Resuming from {current} (existing data up to {existing['datetime'].max()})")
    else:
        current = date(2011, 1, 1)

    final_end = date.today()

    while current < final_end:
        chunk_end = min(current + timedelta(days=27), final_end)
        readings = readdata.fetch_range_robust(current, chunk_end)
        readdata.save_chunk(readings, CSV_PATH)
        current = chunk_end
        time.sleep(0.5)

    print("Done.")


if __name__ == "__main__":                                                                                                         backfill()
