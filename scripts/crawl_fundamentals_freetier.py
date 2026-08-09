"""Collect vnstock fundamentals -> parquet, for the free-tier feasibility
test (08-08-26). Runs in a SEPARATE minimal conda env (`vnfund`), never in
the production env.

WHY A SEPARATE ENV: vnstock 4.0.5 (the version that actually returns
fundamentals -- 4.0.2 returns 0 rows) pins pandas<3, so installing it in
the production env would force pandas 3.0.2 -> 2.3.3 and numpy -> 2.2.6
across the whole ML stack. Verified in a throwaway clone. So: collect here,
write parquet, analyse in the production env. Nothing downstream ever
imports vnstock 4.0.5.

FREE-TIER LIMIT: `Community edition: Financial statements limited to 8
periods.` The point of this crawl is to find out whether those 8 periods
are enough to answer ONE question cheaply, before paying for full history:
does cross-sectional P/E / P/B / ROE rank predict forward 20d returns?

Output: data/fundamentals_freetier.parquet with columns
  ticker, year, quarter, item_en, value
in LONG format (vnstock returns wide, with periods as columns and duplicate
column headers on period="year" -- rows 0/1 carry the true Year/Quarter,
which is the only reliable way to read the period labels).

Run (in the vnfund env):
  <vnfund_python> scripts/crawl_fundamentals_freetier.py [n_tickers]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "fundamentals_freetier.parquet"

# The value/quality line items worth testing. Names must match vnstock's
# `item_en` exactly (verified live 08-08-26 against FPT/VCB/HPG).
WANTED = [
    "P/E", "P/B", "P/S", "Price/Cash Flow", "EV/EBITDA",
    "ROE (%)", "ROA (%)", "Gross Margin (%)", "EBIT Margin (%)",
    "Debt/Equity", "Current Ratio", "Quick Ratio",
    "Market Cap", "Dividend Yield (%)", "Outstanding Shares (mil)",
]


def tickers_from_ohlcv(limit: int | None) -> list[str]:
    """Reuse the local OHLCV shards as the universe -- guarantees every
    fundamentals row can be joined to a price series later."""
    syms = sorted({p.stem.replace("ohlcv_", "").upper()
                   for p in (REPO / "data").glob("ohlcv_*.parquet")})
    return syms[:limit] if limit else syms


def fetch_one(symbol: str) -> pd.DataFrame:
    """One ticker's quarterly ratios, reshaped LONG. Empty frame on failure.

    vnstock returns WIDE: cols = [item, item_en, item_id, <period>, ...] and
    the column HEADERS are unreliable (duplicated on period="year"), so the
    real period labels live in dedicated ROWS instead.

    Those rows are located by `item_id` == 'year' / 'quarter', NOT by
    position: the number and order of metadata rows varies per ticker. An
    earlier version read rows 0 and 1 positionally and produced impossible
    labels like "2021 Q5" for some tickers -- caught by sanity-checking the
    per-ticker period distribution before trusting the crawl. (Note the
    label lives in `item_id`; `item_en` on those metadata rows is just "0".)
    """
    from vnstock import Finance

    fin = Finance(symbol=symbol, source="VCI")
    df = fin.ratio(period="quarter")
    need = {"item_en", "item_id"}
    if df is None or len(df) < 3 or df.shape[1] <= 3 or not need.issubset(df.columns):
        return pd.DataFrame()

    label = df["item_id"].astype(str).str.strip().str.lower()
    yr_rows = df[label == "year"]
    qt_rows = df[label == "quarter"]
    if yr_rows.empty or qt_rows.empty:
        return pd.DataFrame()

    data_cols = list(df.columns[3:])
    years = yr_rows.iloc[0, 3:].tolist()
    quarters = qt_rows.iloc[0, 3:].tolist()

    sub = df[df["item_en"].isin(WANTED)]
    rows: list[dict] = []
    for _, r in sub.iterrows():
        item = str(r["item_en"])
        for col, yr, qt in zip(data_cols, years, quarters, strict=False):
            val = r[col]
            if pd.isna(val):
                continue
            try:
                yr_i, qt_i, val_f = int(float(yr)), int(float(qt)), float(val)
            except (TypeError, ValueError):
                continue
            # Hard sanity gate: anything outside a real calendar quarter or a
            # plausible year means the period labels were mis-read, and a bad
            # period label silently becomes look-ahead bias downstream.
            if not (1 <= qt_i <= 4) or not (2000 <= yr_i <= 2100):
                continue
            rows.append({"ticker": symbol, "year": yr_i, "quarter": qt_i,
                         "item_en": item, "value": val_f})
    return pd.DataFrame(rows)


def fetch_with_retry(symbol: str, attempts: int = 4) -> pd.DataFrame:
    """fetch_one with backoff. The community tier caps at 60 requests/MINUTE
    and one ratio() call costs several requests internally, so an unthrottled
    loop trips the limiter within a few dozen tickers (observed live: the
    crawl died at exit 1 partway through a 359-ticker run)."""
    delay = 6.0
    for a in range(attempts):
        try:
            return fetch_one(symbol)
        except Exception:
            if a == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 1.6
    return pd.DataFrame()


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    syms = tickers_from_ohlcv(limit)
    # Community tier: 60 req/min. Each ticker costs >1 request, so pace well
    # under 1/sec and keep the whole run inside the limiter.
    pace = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5
    print(f"Crawling fundamentals for {len(syms)} tickers "
          f"(free tier, 8 periods each, {pace:.1f}s pace) ...")

    frames: list[pd.DataFrame] = []
    t0 = time.perf_counter()
    fails = 0
    for i, s in enumerate(syms, 1):
        try:
            df = fetch_with_retry(s)
        except Exception as exc:  # noqa: BLE001 -- one bad ticker must not stop the crawl
            fails += 1
            if fails <= 5:
                print(f"  [{i}/{len(syms)}] {s} FAILED: {type(exc).__name__}: {str(exc)[:110]}")
            continue
        if not df.empty:
            frames.append(df)
        if i % 10 == 0 or i == len(syms):
            el = time.perf_counter() - t0
            print(f"  [{i}/{len(syms)}] rows={sum(len(f) for f in frames)}  "
                  f"fails={fails}  {el:.0f}s  ({el / i:.2f}s/ticker)", flush=True)
        # Incremental save: a 15-minute crawl that dies at ticker 300 should
        # not lose everything.
        if frames and i % 50 == 0:
            OUT.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(frames, ignore_index=True).to_parquet(OUT, index=False)
        time.sleep(pace)

    if not frames:
        print("Nothing collected — aborting (no parquet written).")
        return

    out = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\nWrote {len(out)} rows -> {OUT}")
    print(f"  tickers with data : {out['ticker'].nunique()}")
    print(f"  period range      : {out['year'].min()}Q{out.loc[out['year'].idxmin(), 'quarter']} .. "
          f"{out['year'].max()}Q{out.loc[out['year'].idxmax(), 'quarter']}")
    print(f"  distinct periods   : {out.groupby(['year', 'quarter']).ngroups}")
    print(f"  line items         : {sorted(out['item_en'].unique())}")


if __name__ == "__main__":
    main()
