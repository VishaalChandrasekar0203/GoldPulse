from __future__ import annotations

from datetime import date, timedelta

import holidays
import pandas as pd


class IndiaMarketCalendar:
    """Weekdays excluding Indian national holidays (approximation of NSE/MCX)."""

    def __init__(self, years: range | list[int] | None = None):
        years = list(years or range(2009, 2029))
        self._holidays = holidays.country_holidays("IN", years=years)

    def is_trading_day(self, d: date) -> bool:
        if d.weekday() >= 5:
            return False
        return d not in self._holidays

    def next_trading_day(self, d: date) -> date:
        cur = d
        for _ in range(15):
            if self.is_trading_day(cur):
                return cur
            cur += timedelta(days=1)
        return cur

    def trading_days(self, start: date, end: date) -> list[date]:
        days = []
        cur = start
        while cur <= end:
            if self.is_trading_day(cur):
                days.append(cur)
            cur += timedelta(days=1)
        return days

    def add_trading_days(self, d: date, n: int) -> date:
        if n <= 0:
            return d
        cur = d
        added = 0
        while added < n:
            cur += timedelta(days=1)
            if self.is_trading_day(cur):
                added += 1
        return cur

    def trading_days_between(self, start: date, end: date) -> int:
        return len(self.trading_days(start + timedelta(days=1), end))

    def holiday_name(self, d: date) -> str | None:
        name = self._holidays.get(d)
        return str(name) if name else None


def as_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.date
