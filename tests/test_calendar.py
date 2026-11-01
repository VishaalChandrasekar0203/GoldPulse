from __future__ import annotations

from datetime import date

from gold.data.calendar import IndiaMarketCalendar


def test_weekend_is_not_trading_day():
    cal = IndiaMarketCalendar(years=[2024])
    assert not cal.is_trading_day(date(2024, 1, 6))  # Saturday
    assert not cal.is_trading_day(date(2024, 1, 7))  # Sunday
    assert cal.is_trading_day(date(2024, 1, 5))  # Friday


def test_add_trading_days_skips_weekend():
    cal = IndiaMarketCalendar(years=[2024])
    nxt = cal.add_trading_days(date(2024, 1, 5), 1)
    assert nxt == date(2024, 1, 8)
