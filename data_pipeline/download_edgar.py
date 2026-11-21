"""Download reported revenue from SEC EDGAR (XBRL company facts) and write edgar_evidence.csv.

One evidence line = one revenue value reported by one filing for one date range, in the schema of
data/edgar_evidence.csv:
  evidence_id, source, entity, known_at, text, gold_concept, gold_form, gold_fy, gold_fp,
  gold_start, gold_end, gold_value_musd   (+ gold_entity_name, xbrl_concept, known_at_source)

Sources (public, no key; SEC asks for a User-Agent with your name and e-mail and <= 10 requests/s):
  * https://www.sec.gov/files/company_tickers.json             ticker -> CIK
  * https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json  values, forms, fiscal periods, filed date
  * https://data.sec.gov/submissions/CIK##########.json          acceptance time of each accession

Choices that keep the evidence consistent with edgar_rules.md:
  * forms kept: 10-Q, 10-Q/A, 10-K, 10-K/A.
  * ranges kept: 3-month, 6-month and 9-month year-to-date from 10-Qs; fiscal-year totals from 10-Ks.
    3-month figures that some 10-Ks tag for Q4 are dropped (rule 4: Q4 is never filed directly).
  * revenue concept: XBRL filers switch tags over time (SalesRevenueNet -> RevenueFromContract...).
    For each filing, the first available tag in a company-specific priority order is used; the
    order starts with the tag of the company's latest 10-K. All lines are labelled "Revenues".
  * known_at = EDGAR acceptance time (US Eastern, as shown on the filing index). If missing, the
    filing date at 17:30, the latest time a filing can be accepted for that date (never leaks).

  python download_edgar.py --tickers AAPL MSFT NVDA --user-agent "Jane Doe jane@uni.edu" --since 2015-01-01 --out ../data/raw
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd
import requests

CANDIDATES = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
              "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet",
              "SalesRevenueServicesNet", "RevenuesNetOfInterestExpense"]
FORMS = {"10-Q", "10-Q/A", "10-K", "10-K/A"}
COLS = ["evidence_id", "source", "entity", "known_at", "text", "gold_concept", "gold_form", "gold_fy", "gold_fp",
        "gold_start", "gold_end", "gold_value_musd", "gold_entity_name", "xbrl_concept", "known_at_source"]


def dur_class(start, end):
    d = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    return "3M" if 80 <= d <= 120 else "6M" if 170 <= d <= 200 else "9M" if 250 <= d <= 290 else \
        "FY" if 350 <= d <= 380 else "other"


# ----------------------------------------------------------------------------- HTTP
class SEC:
    def __init__(self, user_agent, pause=0.15):
        if "@" not in user_agent or "you@uni.edu" in user_agent or "Your Name" in user_agent:
            raise SystemExit("SEC requires a User-Agent with a name and e-mail, e.g. 'Jane Doe jane@uni.edu'")
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.pause = pause

    def get(self, url):
        for attempt in range(5):
            time.sleep(self.pause)
            r = self.s.get(url, timeout=60)
            if r.status_code == 404:
                return None
            if r.status_code in (403, 429) or r.status_code >= 500:
                time.sleep(2 ** attempt * 2); continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()

    def ticker_map(self):
        j = self.get("https://www.sec.gov/files/company_tickers.json")
        return {v["ticker"].upper(): int(v["cik_str"]) for v in j.values()}

    def fetch(self, cik):
        facts = self.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
        subs = self.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        pages = [subs["filings"]["recent"]] if subs else []
        for f in (subs or {}).get("filings", {}).get("files", []):        # older filings
            pg = self.get(f"https://data.sec.gov/submissions/{f['name']}")
            if pg:
                pages.append(pg)
        return dict(facts=facts, submissions=pages)


# ----------------------------------------------------------------------------- payload -> evidence
def acceptance_times(pages):
    """accession -> 'YYYY-MM-DD HH:MM:SS' (Eastern). The JSON ends in 'Z' but the clock time is the
    Eastern acceptance time shown on the EDGAR filing index; it is used as is."""
    out = {}
    for pg in pages:
        for acc, ts in zip(pg.get("accessionNumber", []), pg.get("acceptanceDateTime", [])):
            if ts:
                out[acc] = ts[:19].replace("T", " ")
    return out


def build(ticker, payload, since=None):
    facts = (payload.get("facts") or {}).get("facts", {}).get("us-gaap", {})
    name = (payload.get("facts") or {}).get("entityName", ticker)
    acc_t = acceptance_times(payload.get("submissions", []))
    rows = []
    for concept in CANDIDATES:
        for u in facts.get(concept, {}).get("units", {}).get("USD", []):
            if u.get("form") in FORMS and u.get("start") and u.get("end"):
                rows.append(dict(u, concept=concept))
    if not rows:
        return pd.DataFrame(columns=COLS), f"{ticker}: no revenue concept found ({', '.join(CANDIDATES[:2])}, ...)"
    df = pd.DataFrame(rows)
    df["dc"] = [dur_class(s, e) for s, e in zip(df.start, df.end)]
    is10k = df.form.str.startswith("10-K")
    df = df[(is10k & (df.dc == "FY")) | (~is10k & df.dc.isin(["3M", "6M", "9M"]))]
    # company-specific concept priority: tag of the latest 10-K first
    k = df[df.form.str.startswith("10-K")].sort_values("filed")
    first = [k.concept.iloc[-1]] if len(k) else []
    prio = {c: i for i, c in enumerate(first + [c for c in CANDIDATES if c not in first])}
    df["prio"] = df.concept.map(prio)
    best = df.groupby("accn").prio.transform("min")
    df = df[df.prio == best]
    df = df.sort_values(["accn", "start", "end"]).drop_duplicates(["accn", "start", "end"], keep="first")
    # timing
    df["known_at"] = [acc_t.get(a) or f"{f} 17:30:00" for a, f in zip(df.accn, df.filed)]
    df["known_at_source"] = ["acceptance" if a in acc_t else "filed_date_17:30" for a in df.accn]
    if since:
        df = df[df.known_at >= since]
    if df.empty:
        return pd.DataFrame(columns=COLS), f"{ticker}: nothing after {since}"
    # comparatives: ranges ending well before the filing's own period
    own_end = df.groupby("accn").end.transform("max")
    df["comparative"] = (pd.to_datetime(own_end) - pd.to_datetime(df.end)).dt.days > 30
    df["musd"] = (df.val.astype(float) / 1e6).round(3)
    df = df.sort_values(["known_at", "accn", "end", "start"]).reset_index(drop=True)

    def money(v):
        return f"{v:,.3f}".rstrip("0").rstrip(".")

    out = pd.DataFrame(dict(
        source="edgar", entity=ticker, known_at=df.known_at,
        text=[f"Form {r.form} (FY{'' if pd.isna(r.fy) else int(r.fy)} {r.fp}) filed {r.filed}: {name} ({ticker}) Revenues for {r.start} to {r.end} "
              f"= ${money(r.musd)}M" + (" (prior-year comparative)" if r.comparative else "")
              + (" (amended filing)" if r.form.endswith("/A") else "") + "."
              for r in df.itertuples()],
        gold_concept="Revenues", gold_form=df.form, gold_fy=df.fy.astype("Int64"), gold_fp=df.fp, gold_start=df.start,
        gold_end=df.end, gold_value_musd=df.musd, gold_entity_name=name, xbrl_concept=df.concept,
        known_at_source=df.known_at_source))
    out.insert(0, "evidence_id", [f"{ticker}-{i:05d}" for i in range(1, len(out) + 1)])
    late = sum(1 for a, f in zip(df.known_at, df.filed) if a[:10] > f)
    msg = (f"{ticker}: {len(out)} lines from {df.accn.nunique()} filings, concepts {sorted(set(df.concept))}, "
           f"{(df.form.str.endswith('/A')).sum()} amended, {(out.known_at_source != 'acceptance').sum()} without acceptance time"
           + (f", WARNING {late} acceptance times after the filing date" if late else ""))
    return out[COLS], msg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="*", default=[])
    p.add_argument("--tickers-file")
    p.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT", ""))
    p.add_argument("--since", default="2015-01-01", help="keep filings accepted on/after this date")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"))
    p.add_argument("--raw-dir", default=None, help="also save raw API payloads here (JSON)")
    a = p.parse_args()
    tickers = [t.upper() for t in a.tickers]
    if a.tickers_file:
        tickers += [x.strip().upper() for x in open(a.tickers_file) if x.strip()]
    if not tickers:
        p.error("give --tickers or --tickers-file")
    sec = SEC(a.user_agent)
    cik = sec.ticker_map()
    frames = []
    for i, t in enumerate(tickers, 1):
        if t.replace(".", "-") not in cik and t not in cik:
            print(f"[{i}/{len(tickers)}] {t}: no CIK", file=sys.stderr); continue
        payload = sec.fetch(cik.get(t, cik.get(t.replace(".", "-"))))
        if a.raw_dir:
            os.makedirs(a.raw_dir, exist_ok=True)
            json.dump(payload, open(f"{a.raw_dir}/edgar_{t}.json", "w"))
        df, msg = build(t, payload, a.since)
        print(f"[{i}/{len(tickers)}] {msg}")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)
    os.makedirs(a.out, exist_ok=True)
    out.to_csv(f"{a.out}/edgar_evidence.csv", index=False)
    print(f"wrote {len(out)} rows for {out.entity.nunique()} companies to {a.out}/edgar_evidence.csv")


if __name__ == "__main__":
    main()
