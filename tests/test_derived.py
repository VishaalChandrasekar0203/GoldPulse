from __future__ import annotations

from datetime import date

from gold.config import TROY_OUNCE_GRAMS
from gold.ingestion.derived import derive_gold_inr_10g
from tests.helpers import make_obs


def test_parity_formula_inner_join_only(store, cfg):
    store.upsert_observations(make_obs("gold_usd_oz", [(date(2024, 1, 2), 2000.0), (date(2024, 1, 3), 2010.0)]), "gold_usd_oz")
    # USDINR missing on 3 Jan — that day must not be fabricated.
    store.upsert_observations(make_obs("usd_inr", [(date(2024, 1, 2), 80.0)]), "usd_inr")
    out = derive_gold_inr_10g(store, cfg, run_id="t")
    assert len(out) == 1
    expected = 2000.0 * 80.0 * (10.0 / TROY_OUNCE_GRAMS)
    assert abs(out["value"].iloc[0] - expected) < 1e-6
    assert out["is_derived"].iloc[0]
