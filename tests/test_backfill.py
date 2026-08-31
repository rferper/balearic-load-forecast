"""The backfill job's resume behaviour, with the network stubbed out."""

from datetime import date, timedelta

from balearic_load_forecast.application import backfill
from balearic_load_forecast.settings import BackfillConfig


def _stub_api(monkeypatch, requested: list) -> None:
    """Record every requested range and return one reading per day."""

    def fake_fetch(start, end):
        requested.append((start, end))
        return [
            {"datetime": f"{start + timedelta(days=n)}T00:00:00+00:00", "value": 100.0}
            for n in range((end - start).days)
        ]

    monkeypatch.setattr(backfill.ree, "fetch_range_robust", fake_fetch)


def test_a_fresh_backfill_starts_at_the_configured_day(tmp_path, monkeypatch):
    # Given: no landing file and a start date one week ago
    requested = []
    _stub_api(monkeypatch, requested)
    start = date.today() - timedelta(days=7)
    config = BackfillConfig(
        raw_csv=tmp_path / "raw.csv", start=start, pause_seconds=0.0
    )

    # When: the backfill runs
    written = backfill.run(config)

    # Then: it fetches from the configured start and writes what it got
    assert requested[0][0] == start
    assert written == 7
    assert config.raw_csv.exists()


def test_a_rerun_resumes_after_the_newest_stored_day(tmp_path, monkeypatch):
    # Given: a landing file already holding data up to four days ago
    requested = []
    _stub_api(monkeypatch, requested)
    last = date.today() - timedelta(days=4)
    path = tmp_path / "raw.csv"
    path.write_text(f"datetime,value\n{last} 00:00:00+00:00,100.0\n", encoding="utf-8")
    config = BackfillConfig(raw_csv=path, pause_seconds=0.0)

    # When: the backfill runs again
    backfill.run(config)

    # Then: it starts the day after the newest stored day, not from 2011
    assert requested[0][0] == last + timedelta(days=1)


def test_an_already_current_backfill_does_nothing(tmp_path, monkeypatch):
    # Given: a landing file already current through today
    requested = []
    _stub_api(monkeypatch, requested)
    path = tmp_path / "raw.csv"
    today = date.today()
    path.write_text(f"datetime,value\n{today} 00:00:00+00:00,100.0\n", encoding="utf-8")

    # When: the backfill runs
    written = backfill.run(BackfillConfig(raw_csv=path, pause_seconds=0.0))

    # Then: no API call is made at all
    assert written == 0
    assert requested == []


def test_chunking_respects_the_configured_width(tmp_path, monkeypatch):
    # Given: a 20-day gap and a 7-day chunk width
    requested = []
    _stub_api(monkeypatch, requested)
    config = BackfillConfig(
        raw_csv=tmp_path / "raw.csv",
        start=date.today() - timedelta(days=20),
        chunk_days=7,
        pause_seconds=0.0,
    )

    # When: the backfill runs
    backfill.run(config)

    # Then: no single request exceeds the configured width
    assert all((end - start).days <= 7 for start, end in requested)
    assert len(requested) == 3
