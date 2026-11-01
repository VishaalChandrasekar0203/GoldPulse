# Indian market data sources: MCX futures and IBJA

**Date.** 2026-11-01  
**Rule.** Do not scrape exchange websites. Do not fabricate prints. Do not relabel international parity (`gold_inr_10g`) as MCX or IBJA. GOLDBEES remains an onshore ETF proxy, not a local-premium series.

This note is a licensing and coverage assessment, not an ingestion. No MCX or IBJA series is stored in DuckDB.

---

## 1. MCX Gold Futures

**What we would need.** Daily (or higher-frequency) gold futures in INR per 10g: contract-wise OHLC, volume, open interest, and preferably a continuous front-month series with a documented roll. Spot polled prices are a separate MCX product.

**Official channel.** Multi Commodity Exchange of India datafeed desk.

| Item | Detail |
|---|---|
| Page | https://www.mcxindia.com/technology/datafeed |
| Product/charges | https://www.mcxindia.com/technology/datafeed/data-feed-product-and-charges |
| Request form | https://www.mcxindia.com/technology/datafeed/online-data-feed-request-form |
| Contact | datafeed@mcxindia.com · +91 22 6649 4040 |
| Agreement | Required before any feed. Redistribution needs prior MCX approval and a redistributor agreement. |

**Products (as published by MCX).**

| Product | Frequency / content | Coverage notes | Indicative cost (ex-tax, domestic vendor list effective 1 Jul 2024) |
|---|---|---|---|
| End-of-day bhavcopy | Daily OHLC, volume, value, OI by contract | Up to **3 months** of history, **excluding the last 12 months**, may be free on request. Paid EOD is listed at ₹2,00,000 / year. | ₹2,00,000 / year (paid EOD) |
| Historical trade book | Tick trades | Not sold for periods **shorter than one year**. Delivered on HDD purchased by the exchange; HDD billed separately. | All commodities ₹2,50,000 / year; single commodity ₹50,000 / year |
| Historical order book | Order-book snapshots | Same one-year minimum and HDD delivery. | All commodities ₹2,50,000 / year; single commodity ₹50,000 / year |
| Historical trade **and** order book | Combined | Same constraints. | All commodities ₹5,00,000 / year; single commodity ₹1,00,000 / year |
| MCX spot (polled) | Eligible commodities | Separate from futures. | ₹2,00,000 / year |
| Real-time L1–L3 / EOBI | Live | Vendor list prices ₹15L–₹54L / year plus per-terminal variable fees. Not required for this research archive. | See MCX domestic vendor tariff |
| Connectivity | Leased line or internet | Required for live feeds, not for HDD history. | Leased line ₹1,50,000 / year; internet ₹1,00,000 / year |

Fees are INR per year, exclusive of tax, and MCX may revise them. Research/internal (“non-display”) use still requires an agreement; it is not a public CSV.

**Authorised vendors.** MCX publishes an authorised realtime vendor list on the same datafeed page. Third-party historical APIs (for example Global Datafeeds GetHistory) are quote-based and still sit under an MCX licence. Retail “historical NSE” SKUs are not a substitute for licensed MCX gold.

**What this project will not do.** Scrape mcxindia.com bhavcopies, reverse-engineer a vendor terminal, or splice COMEX×USDINR into a series named `mcx_gold_futures`.

---

## 2. IBJA / IBJA-IGM rupee gold

**What we would need.** Daily AM/PM rupee gold (and silver) prints with a stable purity definition, plus a bulk history long enough for walk-forward tests (ideally 10+ years).

**Official channels.**

| Item | Detail |
|---|---|
| IBJA Rates API | https://indiagoldratesapi.com and https://www.indiagoldratesapi.com |
| FAQ | https://indiagoldratesapi.com/Faqs.aspx |
| Pricing application | https://www.indiagoldratesapi.com/Pricing.aspx (quote by email; list price is not published) |
| Contacts | nagaraj.iyer@ibja.in · +91 98211 61214; cc Ankitawadke@ibja.in · +91 77188 88007 |
| IBJA-IGM benchmark entity | https://ibja-igm.com/ — daily morning and afternoon 995-purity INR fixings; licensing via the site contact (info@ibja-igm.com per the site) |

**API behaviour (IBJA Rates).**

- Paid subscription, **minimum one year**, non-refundable, **one company per licence** (group licences on request).
- Daily **opening ~12:30 IST** and **closing ~18:30 IST**. No Saturday/Sunday or bank holidays.
- JSON for the current/latest session. The public FAQ states the API **does not** return arbitrary historical date ranges; you persist prints locally going forward.
- Historical data **up to 20 years** is available **on request**, **.xls only**, by writing to nagaraj.iyer@ibja.in.
- After termination you may **retain** stored history for **internal** use. **Resale, redistribution, or publication** of rates or history without written IBJA consent is prohibited.

**What this project will not do.** Call the IBJA endpoint without a licence, scrape dealer websites, or treat GOLDBEES−parity as IBJA.

---

## 3. What we actually model today

| Series | Label | Status |
|---|---|---|
| `gold_inr_10g` | International parity: `GC=F × USDINR × (10/31.1034768)` | Active. **Not** MCX, **not** IBJA. |
| `gold_etf_inr_10g` | NSE GOLDBEES scaled to 10g | Active onshore ETF proxy. Corporate actions can distort the gap. |
| `mcx_gold_futures` | Restricted catalog row | Unavailable. No observations. |
| `ibja_gold` | Restricted catalog row | Unavailable. No observations. |

Parity remains a separately labelled target. Duty/GST/making-charge/local-premium effects belong in the **scenario** overlay, not in the champion forecast.

---

## 4. Recommendation

1. If the product is **international INR gold**, keep `gold_inr_10g` and do not buy MCX/IBJA.
2. If the product is **the price an Indian buyer or hedger actually transacts**, budget for:
   - MCX single-commodity historical **trade book** (gold) as the futures target, plus EOD bhavcopy going forward; and/or
   - IBJA historical XLS (up to 20y) plus a one-year API to extend the series.
3. After licensed data arrives, rebuild features and walk-forward tests on that series. Do not initialize the new target with parity backfill.
4. Until then, promotion rules stay on the parity target: material, Diebold–Mariano-significant OOS improvement versus naive on the latest holdout **and** multiple earlier windows.
