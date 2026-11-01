from __future__ import annotations

from datetime import date

from tests.helpers import make_obs


def test_upsert_deduplicates_and_updates(store):
    s = "usd_inr"
    first = make_obs(s, [(date(2024, 1, 2), 83.1), (date(2024, 1, 3), 83.2), (date(2024, 1, 3), 83.25)])
    stats = store.upsert_observations(first, s)
    assert stats["total"] == 2
    loaded = store.load_series(s)
    assert len(loaded) == 2
    assert float(loaded.loc[loaded["obs_date"] == date(2024, 1, 3), "value"].iloc[0]) == 83.25

    second = make_obs(s, [(date(2024, 1, 4), 83.4), (date(2024, 1, 3), 83.3)])
    stats = store.upsert_observations(second, s)
    assert stats["inserted"] == 1
    assert stats["updated"] == 1
    loaded = store.load_series(s).sort_values("obs_date")
    assert list(loaded["value"]) == [83.1, 83.3, 83.4]


def test_last_observation_date(store):
    s = "dxy"
    store.upsert_observations(make_obs(s, [(date(2024, 2, 1), 100.0)]), s)
    assert store.last_observation_date(s) == date(2024, 2, 1)
    assert store.last_observation_date("missing") is None
