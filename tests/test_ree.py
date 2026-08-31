"""The REE client, including the bisecting retry.

The API is stubbed at the fetch_range boundary; no test touches the network.
"""

from datetime import date

import pytest
import requests

from balearic_load_forecast.io import ree


class _Response:
    """A minimal stand-in for `requests.Response`."""

    def __init__(self, payload: dict, status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status}")

    def json(self) -> dict:
        return self.payload


def _payload(values: list[dict]) -> dict:
    return {"included": [{"attributes": {"values": values}}]}


def test_fetch_range_sends_the_balearic_geo_parameters(monkeypatch):
    # Given: a stubbed transport that records what it was asked for
    seen = {}

    def fake_get(url, params, timeout):
        seen.update(url=url, params=params, timeout=timeout)
        return _Response(_payload([{"datetime": "x", "value": 1.0}]))

    monkeypatch.setattr(ree.requests, "get", fake_get)

    # When: a range is fetched
    readings = ree.fetch_range(date(2023, 1, 1), date(2023, 1, 8))

    # Then: the Balearic system is requested, hourly, with a timeout set
    assert readings == [{"datetime": "x", "value": 1.0}]
    assert seen["params"]["geo_ids"] == 8742
    assert seen["params"]["time_trunc"] == "hour"
    assert seen["params"]["start_date"] == "2023-01-01T00:00"
    assert seen["timeout"] == ree.TIMEOUT_SECONDS


def test_fetch_range_raises_when_the_api_rejects_the_range(monkeypatch):
    # Given: an API returning 502
    monkeypatch.setattr(ree.requests, "get", lambda *_, **__: _Response({}, status=502))
    # When/Then: the error surfaces rather than being swallowed
    with pytest.raises(requests.HTTPError):
        ree.fetch_range(date(2023, 1, 1), date(2023, 1, 8))


def test_robust_fetch_bisects_a_range_the_api_refuses(monkeypatch):
    # Given: an API that refuses any range wider than two days
    calls = []

    def fake_fetch(start, end):
        calls.append((start, end))
        if (end - start).days > 2:
            raise requests.HTTPError("too wide")
        return [{"datetime": str(start), "value": 1.0}]

    monkeypatch.setattr(ree, "fetch_range", fake_fetch)

    # When: an eight-day range is fetched robustly
    readings = ree.fetch_range_robust(date(2023, 1, 1), date(2023, 1, 9))

    # Then: it splits until every sub-range is accepted, losing nothing
    assert len(readings) == 4  # four two-day chunks
    assert calls[0] == (date(2023, 1, 1), date(2023, 1, 9))  # the wide try first


def test_a_single_day_that_still_fails_is_skipped_not_fatal(monkeypatch):
    # Given: an API with a genuine one-day hole in the middle
    hole = date(2023, 1, 3)

    def fake_fetch(start, end):
        if start <= hole < end:
            raise requests.HTTPError("no data")
        return [{"datetime": str(start), "value": 1.0}]

    monkeypatch.setattr(ree, "fetch_range", fake_fetch)

    # When: a range spanning the hole is fetched
    readings = ree.fetch_range_robust(date(2023, 1, 1), date(2023, 1, 5))

    # Then: the surrounding days survive - one bad day does not abort a backfill
    assert len(readings) > 0
    assert all(str(hole) != r["datetime"] for r in readings)
