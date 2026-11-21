"""Offline end-to-end test: mock API payloads (in the real FRED / SEC / yfinance formats) that encode
the three sample histories -> downloader build() -> question templates -> gold must equal the
hand-checked sample gold in ../data/*_questions.csv.

  python tests/test_offline.py          (from real_data/)
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import download_alfred as DA      # noqa: E402
import download_broker as DB      # noqa: E402
import download_edgar as DE       # noqa: E402
import generate_questions as G    # noqa: E402

SAMPLE = os.path.join(HERE, "samples")


# ----------------------------------------------------------------------------- ALFRED (FRED API format)
def alfred_payload():
    V = ["2026-01-29", "2026-02-26", "2026-03-26", "2026-04-29", "2026-05-28", "2026-06-25", "2026-07-30",
         "2026-08-27", "2026-09-24", "2026-10-29"]
    spans = {  # observation date -> [(realtime_start, realtime_end, value)]
        "2025-10-01": [("2026-01-29", "2026-02-25", "2.5"), ("2026-02-26", "2026-03-25", "2.4"),
                       ("2026-03-26", "2026-09-23", "2.0"), ("2026-09-24", "9999-12-31", "2.3")],
        "2026-01-01": [("2026-04-29", "2026-05-27", "1.8"), ("2026-05-28", "2026-06-24", "1.6"),
                       ("2026-06-25", "2026-09-23", "1.5"), ("2026-09-24", "9999-12-31", "1.9")],
        "2026-04-01": [("2026-07-30", "2026-08-26", "2.4"), ("2026-08-27", "2026-09-23", "2.6"),
                       ("2026-09-24", "9999-12-31", "2.1")],
        "2026-07-01": [("2026-10-29", "9999-12-31", "1.2")],
    }
    obs = [dict(realtime_start=a, realtime_end=b, date=d, value=v) for d, ss in spans.items() for a, b, v in ss]
    obs.append(dict(realtime_start="2026-01-29", realtime_end="2026-02-25", date="2025-07-01", value="."))
    meta = dict(id="ZETA", title="Zeta real GDP", frequency_short="Q", units="Percent Change", units_short="% SAAR")
    return dict(meta=meta, observations=obs, vintage_dates=V)


def test_alfred():
    cfg = dict(DA.SERIES["A191RL1Q225SBEA"], entity="ZETA_GDP", label="Zeta's real GDP growth")
    ev = DA.build("ZETA", alfred_payload(), cfg)
    assert list(ev.gold_release) == ["advance", "second", "third", "advance", "second", "third", "advance", "second",
                                     "annual update", "annual update", "third", "advance"], list(ev.gold_release)
    return ev


# ----------------------------------------------------------------------------- EDGAR (companyfacts + submissions)
def edgar_payload():
    S = [("2025-05-08", "10-Q", 2025, "Q1", "2025-01-01", "2025-03-31", 980),
         ("2025-08-07", "10-Q", 2025, "Q2", "2025-04-01", "2025-06-30", 1020),
         ("2025-08-07", "10-Q", 2025, "Q2", "2025-01-01", "2025-06-30", 2000),
         ("2025-11-06", "10-Q", 2025, "Q3", "2025-07-01", "2025-09-30", 1050),
         ("2025-11-06", "10-Q", 2025, "Q3", "2025-01-01", "2025-09-30", 3050),
         ("2026-02-26", "10-K", 2025, "FY", "2025-01-01", "2025-12-31", 4120),
         ("2026-02-26", "10-K", 2025, "FY", "2025-10-01", "2025-12-31", 1070),    # Q4 tagged in 10-K: must be dropped
         ("2026-05-07", "10-Q", 2026, "Q1", "2026-01-01", "2026-03-31", 1060),
         ("2026-05-07", "10-Q", 2026, "Q1", "2025-01-01", "2025-03-31", 980),
         ("2026-06-18", "10-K/A", 2025, "FY", "2025-01-01", "2025-12-31", 4060),
         ("2026-08-06", "10-Q", 2026, "Q2", "2026-04-01", "2026-06-30", 1110),
         ("2026-08-06", "10-Q", 2026, "Q2", "2026-01-01", "2026-06-30", 2170),
         ("2026-08-06", "10-Q", 2026, "Q2", "2025-04-01", "2025-06-30", 1020),
         ("2026-11-05", "10-Q", 2026, "Q3", "2026-07-01", "2026-09-30", 1130),
         ("2026-11-05", "10-Q", 2026, "Q3", "2026-01-01", "2026-09-30", 3300),
         ("2026-11-05", "10-Q", 2026, "Q3", "2025-07-01", "2025-09-30", 1050)]
    acc = {d: f"0000000000-{d[2:4]}-{i:06d}" for i, d in enumerate(sorted({s[0] for s in S}), 1)}
    units = [dict(start=s, end=e, val=v * 1_000_000, accn=acc[d], fy=fy, fp=fp, form=f, filed=d) for d, f, fy, fp, s, e, v in S]
    # an older concept reported alongside in the first filing only: must lose to the priority concept
    old = [dict(units[0], val=1)]
    facts = dict(cik=1, entityName="Borealis Corp", facts={"us-gaap": {
        "Revenues": {"units": {"USD": units}}, "SalesRevenueNet": {"units": {"USD": old}}}})
    times = {"2026-06-18": "16:30:00"}
    sub = dict(accessionNumber=list(acc.values()),
               acceptanceDateTime=[f"{d}T{times.get(d, '16:05:00')}.000Z" for d in acc],
               form=["x"] * len(acc), filingDate=list(acc))
    return dict(facts=facts, submissions=[sub])


def test_edgar():
    ev, msg = DE.build("BRLS", edgar_payload())
    print(" ", msg)
    assert len(ev) == 15 and (ev.known_at_source == "acceptance").all()
    assert ev[ev.gold_form == "10-K/A"].known_at.iloc[0] == "2026-06-18 16:30:00"
    return ev


# ----------------------------------------------------------------------------- broker (yfinance format)
def broker_payload():
    N = [("2025-02-03 10:15:00", "Delta Partners", "Hold", "Hold", "main", "Maintains", 90.0, None),
         ("2026-01-12 09:40:00", "Atlas Securities", "Buy", None, "init", "Announces", 120.0, None),
         ("2026-06-15 11:20:00", "Birch Capital", "Outperform", "Outperform", "main", "Raises", 55.0, 50.0),
         ("2026-07-28 07:05:00", "Cobalt & Co.", "Neutral", "Overweight", "down", "Lowers", 52.0, 60.0),
         ("2026-07-28 16:40:00", "Birch Capital", "Outperform", "Outperform", "main", "Lowers", 56.0, 60.0),
         ("2026-09-09 13:00:00", "Atlas Securities", "Buy", "Buy", "main", "Raises", 66.0, 60.0),
         ("2026-09-20 12:30:00", "Atlas Securities", "Buy", "Buy", "reit", "Maintains", 66.0, 66.0),
         ("2026-10-21 08:10:00", "TD Cobalt", "Overweight", "Neutral", "up", "Raises", 64.0, 52.0)]
    notes = pd.DataFrame(N, columns=["GradeDate", "Firm", "ToGrade", "FromGrade", "Action", "priceTargetAction",
                                     "currentPriceTarget", "priorPriceTarget"])
    notes["GradeDate"] = pd.to_datetime(notes.GradeDate)
    notes["ticker"] = "AUR"
    splits = pd.DataFrame(dict(ticker=["AUR"], date=["2026-06-02"], ratio=[2.0]))
    renames = pd.DataFrame(dict(old_name=["Cobalt & Co."], new_name=["TD Cobalt"], announced_at=["2026-09-01 09:00:00"]))
    return notes, splits, renames


def test_broker():
    notes, splits, renames = broker_payload()
    ev = DB.build(notes, splits, renames, since=None, tz_in="America/New_York", tz_out="America/New_York")
    assert list(ev.gold_event).count("rename") == 1 and list(ev.gold_event).count("split") == 1
    print("  e.g.", ev.text.iloc[5])
    return ev


# ----------------------------------------------------------------------------- compare with the sample gold
def check(src, ev):
    Eng, templates = G.SOURCES[src]
    eng = Eng(ev)
    ref = pd.read_csv(f"{SAMPLE}/{src}_questions.csv", keep_default_na=False, dtype=str)
    got = {}
    for i, q in enumerate(templates(eng, G.SAMPLE_ASOF[src], G.SAMPLE_TODAY), 1):
        for r in G.rows_for(src, eng, q, 1, i, G.SAMPLE_ASOF[src], G.SAMPLE_TODAY):
            got[(r["qtype"], r["mode"])] = r["gold"]
    bad = []
    for r in ref.itertuples():
        g = got.get((r.qtype, r.mode))
        ok = g is not None and (g == r.gold if CANNOT in (g, r.gold) or r.answer_type != "number"
                                else abs(float(g) - float(r.gold)) <= float(r.tol) + 1e-9)
        if not ok:
            bad.append((r.qtype, r.mode, r.gold, g))
    print(f"{src}: downloader -> generator gold matches sample for {len(ref) - len(bad)}/{len(ref)}" + (f"  {bad}" if bad else ""))
    return not bad


CANNOT = "CANNOT_ANSWER"

if __name__ == "__main__":
    ok = all([check("alfred", test_alfred()), check("edgar", test_edgar()), check("broker", test_broker())])
    sys.exit(0 if ok else 1)
