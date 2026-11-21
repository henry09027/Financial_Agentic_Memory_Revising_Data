"""Download macro data vintages from ALFRED (St. Louis Fed) and write alfred_evidence.csv.

One evidence line = one published estimate of one period (quarter or month), stamped with its
release time, in the same schema as data/alfred_evidence.csv:
  evidence_id, source, entity, known_at, text, gold_series, gold_period, gold_release, gold_value
  (+ gold_label, gold_unit: used only to word the questions)

Which releases become evidence lines (rules card, rule 4: "each later release counts, even if the
value is unchanged"):
  * the first print (advance estimate),
  * the next two scheduled releases after it (second and third estimates), even if unchanged,
  * every later release that changed the value (annual / benchmark updates).
ALFRED does not say whether an annual update re-estimated a period without changing it, so an
unchanged annual update is not an evidence line.

known_at = vintage date + the series' release time (08:30 for BEA GDP and BLS payrolls), local US
Eastern time, as in the samples.

Needs a free FRED API key: https://fredaccount.stlouisfed.org/apikey
  export FRED_API_KEY=...
  python download_alfred.py --series A191RL1Q225SBEA PAYEMS --start 2005-01-01 --out ../data/raw
"""
from __future__ import annotations

import argparse
import json
import os
import time
from bisect import bisect_right

import pandas as pd
import requests

API = "https://api.stlouisfed.org/fred"
ALL_TIME = dict(realtime_start="1776-07-04", realtime_end="9999-12-31")

# Known series: wording for evidence text and questions. Others fall back to FRED metadata.
SERIES = {
    "A191RL1Q225SBEA": dict(
        entity="US_GDP", label="US real GDP growth", unit="percent, annual rate", short="% SAAR", time="08:30:00",
        first="{lab}: US real GDP {dir} at an annual rate of {absv}% in {per}.",
        later="{lab}: US real GDP growth for {per} is now estimated at an annual rate of {v}%."),
    "PAYEMS": dict(
        entity="US_PAYROLLS", label="US nonfarm payroll employment", unit="thousands of persons", short="thous.",
        time="08:30:00",
        first="{lab}: US nonfarm payroll employment was {v} thousand in {per}.",
        later="{lab}: US nonfarm payroll employment for {per} is now estimated at {v} thousand."),
    "UNRATE": dict(
        entity="US_UNRATE", label="US unemployment rate", unit="percent", short="%", time="08:30:00",
        first="{lab}: the US unemployment rate was {v}% in {per}.",
        later="{lab}: the US unemployment rate for {per} is now estimated at {v}%."),
    "CPIAUCSL": dict(
        entity="US_CPI", label="US CPI (all urban consumers, SA index)", unit="index 1982-84=100", short="index",
        time="08:30:00",
        first="{lab}: the US CPI (SA) stood at {v} in {per}.",
        later="{lab}: the US CPI (SA) for {per} is now estimated at {v}."),
}


# ----------------------------------------------------------------------------- HTTP
def get(path, api_key, **params):
    params.update(api_key=api_key, file_type="json")
    for attempt in range(5):
        r = requests.get(f"{API}/{path}", params=params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt); continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def fetch(series_id, api_key, start):
    """Raw API payloads for one series: metadata, real-time observations, vintage dates."""
    meta = get("series", api_key, series_id=series_id)["seriess"][0]
    obs, off = [], 0
    while True:
        j = get("series/observations", api_key, series_id=series_id, observation_start=start,
                limit=100000, offset=off, **ALL_TIME)
        obs += j["observations"]; off += len(j["observations"])
        if off >= int(j.get("count", 0)) or not j["observations"]:
            break
    vint, off = [], 0
    while True:
        j = get("series/vintagedates", api_key, series_id=series_id, limit=10000, offset=off, **ALL_TIME)
        vint += j["vintage_dates"]; off += len(j["vintage_dates"])
        if off >= int(j.get("count", 0)) or not j["vintage_dates"]:
            break
    return dict(meta=meta, observations=obs, vintage_dates=vint)


# ----------------------------------------------------------------------------- payload -> evidence
def period_label(date, freq):
    d = pd.Timestamp(date)
    return f"{d.year}Q{(d.month - 1) // 3 + 1}" if freq == "Q" else f"{d:%Y-%m}"


def release_label(i, vdate, freq, scheduled):
    """Name of the i-th release of a period (0 = first print)."""
    if freq == "Q":
        names = ["Advance estimate", "Second estimate", "Third estimate"]
    else:
        names = ["First estimate", "Second estimate", "Third estimate"]
    if i < 3 and scheduled:
        return names[i]
    m = pd.Timestamp(vdate).month
    if freq == "Q":
        return "Annual update" if m in (7, 8, 9) else "Revision"
    return "Benchmark revision" if m in (1, 2) else "Revision"


def num(v):
    s = f"{float(v):,.4f}".rstrip("0").rstrip(".")
    return s


def build(series_id, payload, cfg=None):
    meta, obs, vint = payload["meta"], payload["observations"], sorted(payload["vintage_dates"])
    freq = meta.get("frequency_short", "Q")[0]
    cfg = dict(cfg or SERIES.get(series_id) or dict(
        entity=series_id, label=meta.get("title", series_id), unit=meta.get("units", ""),
        short=meta.get("units_short", ""), time="08:30:00",
        first="{lab}: " + meta.get("title", series_id) + " was {v} ({unit}) in {per}.",
        later="{lab}: " + meta.get("title", series_id) + " for {per} is now estimated at {v} ({unit})."))

    by_date = {}
    for o in obs:
        if o["value"] in (".", ""):
            continue
        by_date.setdefault(o["date"], []).append((o["realtime_start"], o["realtime_end"], float(o["value"])))

    rows = []
    for date, spans in sorted(by_date.items()):
        spans.sort()

        def value_at(v):
            for rs, re_, val in spans:
                if rs <= v <= re_:
                    return val
            return None

        first = spans[0][0]
        k = bisect_right(vint, first)
        scheduled = [v for v in vint[k:k + 2] if value_at(v) is not None]          # second + third estimates
        changes = [rs for rs, _, _ in spans[1:]]                                    # every value change
        rels = sorted(set([first] + scheduled + changes))
        for i, v in enumerate(rels):
            val = value_at(v)
            lab = release_label(i, v, freq, scheduled=(i == 0 or v in scheduled))
            per = period_label(date, freq)
            tmpl = cfg["first"] if i == 0 else cfg["later"]
            text = tmpl.format(lab=lab, per=per, v=num(val), absv=num(abs(val)), unit=cfg["unit"],
                               dir="increased" if val >= 0 else "decreased")
            rows.append(dict(source="alfred", entity=cfg["entity"], known_at=f"{v} {cfg['time']}", text=text,
                             gold_series=f"{cfg['label']}, {cfg['short']}", gold_period=per,
                             gold_release={"advance estimate": "advance", "second estimate": "second",
                                           "third estimate": "third"}.get(lab.lower(), lab.lower()),
                             gold_value=val, gold_label=cfg["label"], gold_unit=cfg["unit"]))
    df = pd.DataFrame(rows).sort_values(["known_at", "gold_period"], kind="stable").reset_index(drop=True)
    df.insert(0, "evidence_id", [f"{cfg['entity']}-{i:05d}" for i in range(1, len(df) + 1)])
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--series", nargs="+", default=["A191RL1Q225SBEA"], help="FRED/ALFRED series ids")
    p.add_argument("--start", default="2005-01-01", help="first observation date (period) to keep")
    p.add_argument("--api-key", default=os.environ.get("FRED_API_KEY"))
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"), help="output folder")
    p.add_argument("--raw-dir", default=None, help="also save raw API payloads here (JSON), for reproducibility")
    a = p.parse_args()
    if not a.api_key:
        p.error("set FRED_API_KEY or pass --api-key (free at https://fredaccount.stlouisfed.org/apikey)")
    os.makedirs(a.out, exist_ok=True)
    frames = []
    for sid in a.series:
        payload = fetch(sid, a.api_key, a.start)
        if a.raw_dir:
            os.makedirs(a.raw_dir, exist_ok=True)
            json.dump(payload, open(f"{a.raw_dir}/alfred_{sid}.json", "w"))
        df = build(sid, payload)
        print(f"{sid}: {df.gold_period.nunique()} periods, {len(df)} releases "
              f"({(df.gold_release.str.contains('annual|benchmark|revision')).sum()} later revisions)")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(f"{a.out}/alfred_evidence.csv", index=False)
    print(f"wrote {len(out)} rows to {a.out}/alfred_evidence.csv")


if __name__ == "__main__":
    main()
