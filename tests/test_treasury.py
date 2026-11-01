from __future__ import annotations

from gold.ingestion.treasury_source import parse_real_yield_csv


def test_parse_archive_header():
    text = "date,5 yr,7 yr,10 yr,20 yr,30 yr\n01/02/2003,1.75,2.24,2.43,,\n01/03/2003,1.80,2.20,2.50,2.9,3.1\n"
    df = parse_real_yield_csv(text)
    assert len(df) == 2
    assert float(df.iloc[1]["value"]) == 2.50


def test_parse_headerless_yearly():
    text = "08/28/2026,2.18,2.29,2.42,2.76,2.96\n08/27/2026,2.07,2.19,2.34,2.71,2.92\n"
    df = parse_real_yield_csv(text)
    assert len(df) == 2
    assert float(df.iloc[0]["value"]) == 2.42
