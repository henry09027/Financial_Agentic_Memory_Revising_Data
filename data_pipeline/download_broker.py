"""Download broker actions (ratings + price targets) and splits from Yahoo Finance; write broker_evidence.csv.

One evidence line per broker note, stock split, or broker rename, in the schema of
data/broker_evidence.csv:
  evidence_id, source, entity, known_at, gold_event, gold_firm, gold_action, gold_from_grade,
  gold_to_grade, gold_pt_action, gold_prior_pt, gold_current_pt, text, gold_split_ratio, gold_new_name

Sources
  * notes:  yfinance Ticker.upgrades_downgrades (needs a recent yfinance with the
            priceTargetAction / currentPriceTarget / priorPriceTarget columns), or --input-csv with the
            same columns plus `ticker` (e.g. your existing 43,856-note file).
  * splits: yfinance Ticker.splits (or --splits-csv: ticker,date,ratio).
            known_at = split date 06:00 (effective before the open), as in the samples.
  * renames: Yahoo has no rename feed. broker_renames.csv lists old_name,new_name[,announced_at].
            A rename line is created for every ticker covered under both names. Without
            announced_at, known_at = 1 second before the first note published under the new name
            (across all downloaded tickers). Run with --suggest-renames to list candidate pairs.

Text: each note is rewritten into a sentence like the samples. The price target is stated
"from $X" only when the note raises or lowers it, so missed revisions (rule 6) are visible exactly
when the rules card says they are.

Licence: Yahoo's terms restrict redistribution. Publish this script, not its output.

  pip install "yfinance>=0.2.54"
  python download_broker.py --tickers AAPL MSFT NVDA --since 2016-01-01 --out ../data/raw
  python download_broker.py --input-csv my_notes.csv --out ../data/raw        # no Yahoo calls for notes
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import Counter

import pandas as pd

ACTION = {"init": "initiate", "main": "maintain", "up": "upgrade", "down": "downgrade", "reit": "reiterate"}
NOTE_COLS = ["GradeDate", "Firm", "ToGrade", "FromGrade", "Action", "priceTargetAction", "currentPriceTarget",
             "priorPriceTarget"]
COLS = ["evidence_id", "source", "entity", "known_at", "gold_event", "gold_firm", "gold_action", "gold_from_grade",
        "gold_to_grade", "gold_pt_action", "gold_prior_pt", "gold_current_pt", "text", "gold_split_ratio", "gold_new_name"]


# ----------------------------------------------------------------------------- fetching
def fetch_yahoo(tickers, pause=1.0):
    import yfinance as yf
    notes, splits = [], []
    for i, t in enumerate(tickers, 1):
        for attempt in range(4):
            try:
                tk = yf.Ticker(t)
                ud = tk.upgrades_downgrades
                sp = tk.splits
                break
            except Exception as e:                            # rate limits / transient errors
                print(f"  {t}: {e} (retry {attempt + 1})", file=sys.stderr); time.sleep(5 * (attempt + 1))
        else:
            print(f"  {t}: giving up", file=sys.stderr); continue
        if ud is not None and len(ud):
            ud = ud.reset_index()
            missing = [c for c in NOTE_COLS if c not in ud.columns]
            if missing:
                print(f"  {t}: yfinance returned no {missing}; upgrade yfinance", file=sys.stderr)
            ud["ticker"] = t
            notes.append(ud)
        if sp is not None and len(sp):
            splits.append(pd.DataFrame({"ticker": t, "date": [d.strftime("%Y-%m-%d") for d in sp.index],
                                        "ratio": sp.values.astype(float)}))
        print(f"[{i}/{len(tickers)}] {t}: {0 if ud is None else len(ud)} notes, {0 if sp is None else len(sp)} splits")
        time.sleep(pause)
    n = pd.concat(notes, ignore_index=True) if notes else pd.DataFrame(columns=NOTE_COLS + ["ticker"])
    s = pd.concat(splits, ignore_index=True) if splits else pd.DataFrame(columns=["ticker", "date", "ratio"])
    return n, s


# ----------------------------------------------------------------------------- notes -> evidence
def money(x):
    return f"${float(x):,.2f}"


def clean_pt(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(x) or x <= 0 else x


def clean_grade(x):
    return "" if x is None or pd.isna(x) else str(x).strip()


def note_text(T, firm, act, frm, to, pt_act, prior, cur):
    """Rewrite one Yahoo row into a sentence like the samples. Returns (text, pt_action, prior_stated)."""
    # normalise the price-target action from the numbers when both are known
    if prior is not None and cur is not None:
        pt_act = "Raises" if cur > prior + 1e-9 else "Lowers" if cur < prior - 1e-9 else "Maintains"
    elif cur is not None and pt_act not in ("Raises", "Lowers", "Announces"):
        pt_act = "Maintains"
    elif cur is None:
        pt_act = ""
    if act == "initiate":
        head = f"{firm} initiated coverage of {T}" + (f" with a {to} rating" if to else "")
        joiner = " and"
    elif act in ("upgrade", "downgrade"):
        head = f"{firm} {act}d {T}" + (f" from {frm} to {to}" if frm and to else f" to {to}" if to else "")
        joiner = ", and"
    elif act == "reiterate":
        head = f"{firm} reiterated {to + ' on ' if to else 'its view on '}{T}"
        joiner = ", and"
    else:
        head = f"{firm} maintained {to + ' on ' if to else 'coverage of '}{T}"
        joiner = ", and"
    stated_prior = None
    if cur is None:
        tail = "."
    elif pt_act in ("Raises", "Lowers") and prior is not None:
        tail = f"{joiner} {'raised' if pt_act == 'Raises' else 'lowered'} its price target from {money(prior)} to {money(cur)}."
        stated_prior = prior
    elif pt_act in ("Raises", "Lowers"):
        tail = f"{joiner} {'raised' if pt_act == 'Raises' else 'lowered'} its price target to {money(cur)}."
    elif act == "initiate":
        tail = f" and a {money(cur)} price target."
    else:
        tail = f", with a {money(cur)} price target."
    return head + tail, pt_act, stated_prior


def to_local(ts, tz_in, tz_out):
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize(tz_in)
    return ts.tz_convert(tz_out).tz_localize(None)


def build(notes, splits, renames, since=None, tz_in="UTC", tz_out="America/New_York"):
    rows = []
    notes = notes.drop_duplicates(subset=["ticker"] + [c for c in NOTE_COLS if c in notes.columns])
    for r in notes.itertuples(index=False):
        d = r._asdict()
        t = to_local(d["GradeDate"], tz_in, tz_out)
        firm = str(d["Firm"]).strip()
        act = ACTION.get(str(d.get("Action", "")).strip().lower(), "maintain")
        frm, to = clean_grade(d.get("FromGrade")), clean_grade(d.get("ToGrade"))
        cur, prior = clean_pt(d.get("currentPriceTarget")), clean_pt(d.get("priorPriceTarget"))
        text, pt_act, stated_prior = note_text(d["ticker"], firm, act, frm, to, str(d.get("priceTargetAction") or ""),
                                               prior, cur)
        rows.append(dict(entity=d["ticker"], known_at=t, gold_event="broker_action", gold_firm=firm, gold_action=act,
                         gold_from_grade=frm or None, gold_to_grade=to or None, gold_pt_action=pt_act or None,
                         gold_prior_pt=stated_prior, gold_current_pt=cur, text=text))
    for s in splits.itertuples(index=False):
        k = float(s.ratio)
        if k <= 0 or abs(k - 1) < 1e-9:
            continue
        t = pd.Timestamp(f"{s.date} 06:00:00")
        if k >= 1:
            desc = f"a {fmt_ratio(k)}-for-1 stock split"
            eff = f"Per-share prices from this date are 1/{fmt_ratio(k)} of pre-split prices."
        else:
            desc = f"a 1-for-{fmt_ratio(1 / k)} reverse stock split"
            eff = f"Per-share prices from this date are {fmt_ratio(1 / k)} times pre-split prices."
        rows.append(dict(entity=s.ticker, known_at=t, gold_event="split", gold_split_ratio=k,
                         text=f"Corporate action: {s.ticker} completed {desc} effective before the open on {s.date}. {eff}"))
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=COLS)
    # renames: one line per ticker covered under both names
    notes_df = df[df.gold_event == "broker_action"]
    first_new = notes_df.groupby("gold_firm").known_at.min()
    ren_rows = []
    for rn in renames.itertuples(index=False):
        old, new = rn.old_name.strip(), rn.new_name.strip()
        if new not in first_new.index:
            continue
        when = pd.Timestamp(rn.announced_at) if getattr(rn, "announced_at", None) and pd.notna(rn.announced_at) \
            else first_new[new] - pd.Timedelta(seconds=1)
        for T, g in notes_df.groupby("entity"):
            if old in set(g.gold_firm) and new in set(g.gold_firm):
                ren_rows.append(dict(entity=T, known_at=when, gold_event="rename", gold_firm=old, gold_new_name=new,
                                     text=f"Broker update: {old} now publishes research under the name {new}. "
                                          "Coverage continues unchanged."))
    if ren_rows:
        df = pd.concat([df, pd.DataFrame(ren_rows)], ignore_index=True)
    if since:
        keep_splits = df.gold_event == "split"        # a split before `since` still adjusts later targets
        df = df[(df.known_at >= pd.Timestamp(since)) | keep_splits]
    df["source"] = "broker"
    df = df.sort_values(["entity", "known_at", "gold_event"], kind="stable").reset_index(drop=True)
    df["evidence_id"] = df.groupby("entity").cumcount().add(1).map("{:05d}".format).radd(df.entity + "-")
    df["known_at"] = df.known_at.dt.strftime("%Y-%m-%d %H:%M:%S")
    for c in COLS:
        if c not in df:
            df[c] = None
    return df[COLS]


def fmt_ratio(k):
    return f"{k:.4f}".rstrip("0").rstrip(".")


# ----------------------------------------------------------------------------- diagnostics
GENERIC = {"capital", "securities", "partners", "group", "research", "markets", "bank", "financial", "co", "and",
           "llc", "inc", "the", "company", "advisors", "equity", "investment", "global"}


def suggest_renames(ev, top=30):
    n = ev[ev.gold_event == "broker_action"].copy()
    n["t"] = pd.to_datetime(n.known_at)
    span = n.groupby("gold_firm").t.agg(["min", "max", "count"])
    toks = {f: {w for w in re.findall(r"[a-z]+", f.lower()) if len(w) > 3 and w not in GENERIC} for f in span.index}
    out = []
    for a in span.index:
        for b in span.index:
            if a != b and toks[a] & toks[b] and span.loc[a, "max"] <= span.loc[b, "min"] + pd.Timedelta(days=120):
                out.append((a, b, f"{span.loc[a, 'max']:%Y-%m-%d}", f"{span.loc[b, 'min']:%Y-%m-%d}",
                            int(span.loc[a, "count"]), int(span.loc[b, "count"])))
    out.sort(key=lambda x: -(x[4] + x[5]))
    return pd.DataFrame(out[:top], columns=["old_name", "new_name", "old_last_note", "new_first_note", "n_old", "n_new"])


def diagnostics(ev):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pit_engines import BrokerEngine
    n_notes = n_inf = 0
    ratios = []
    for T, g in ev.groupby("entity"):
        eng = BrokerEngine(g)
        st = eng.state(eng.ev.t.max())
        n_notes += sum(len(v) for v in st.notes.values())
        n_inf += sum(1 for v in st.targets.values() for x in v if x.kind == "inferred")
        # split check: stated "from $X" in the first note after a split vs the last pre-split target
        raw = eng.state(eng.ev.t.max(), adjust_splits=False, use_inferred=False)
        for s in eng.ev[eng.ev.gold_event == "split"].itertuples():
            for f, ns in raw.notes.items():
                after = [n for n in ns if n.t > s.t and pd.notna(n.gold_prior_pt)][:1]
                before = [x for x in raw.targets.get(f, []) if x.eff < s.t]
                if after and before and after[0].t - s.t < pd.Timedelta(days=180):
                    ratios.append((before[-1].value / float(after[0].gold_prior_pt), float(s.gold_split_ratio)))
    print(f"diagnostics: {n_notes} notes, {n_inf} inferred missed revisions ({n_inf / max(n_notes, 1):.0%} of notes)")
    if ratios:
        hits = sum(abs(r / k - 1) < 0.15 for r, k in ratios)
        print(f"split check: {hits}/{len(ratios)} post-split 'from $X' values are ~1/ratio of the pre-split target "
              "(expected if Yahoo targets are as published, i.e. NOT already split-adjusted)")


# ----------------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="*", default=[])
    p.add_argument("--tickers-file", help="one ticker per line")
    p.add_argument("--input-csv", help="existing notes file (yfinance columns + ticker); skips Yahoo for notes")
    p.add_argument("--splits-csv", help="ticker,date,ratio; skips Yahoo for splits")
    p.add_argument("--renames", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "broker_renames.csv"))
    p.add_argument("--since", default="2016-01-01", help="drop notes before this date (splits are always kept)")
    p.add_argument("--grade-tz", default="UTC", help="time zone of naive GradeDate values (yfinance: UTC)")
    p.add_argument("--out-tz", default="America/New_York")
    p.add_argument("--pause", type=float, default=1.0, help="seconds between Yahoo requests")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"))
    p.add_argument("--suggest-renames", action="store_true", help="print candidate broker renames and exit")
    a = p.parse_args()

    tickers = list(a.tickers)
    if a.tickers_file:
        tickers += [x.strip().upper() for x in open(a.tickers_file) if x.strip()]
    if a.input_csv:
        notes = pd.read_csv(a.input_csv)
        tickers = tickers or sorted(notes.ticker.unique())
        splits = pd.read_csv(a.splits_csv) if a.splits_csv else fetch_yahoo(tickers, a.pause)[1]
    else:
        if not tickers:
            p.error("give --tickers, --tickers-file or --input-csv")
        notes, splits = fetch_yahoo(tickers, a.pause)
        if a.splits_csv:
            splits = pd.read_csv(a.splits_csv)
    renames = pd.read_csv(a.renames, comment="#") if os.path.exists(a.renames) else \
        pd.DataFrame(columns=["old_name", "new_name", "announced_at"])

    ev = build(notes, splits, renames, a.since, a.grade_tz, a.out_tz)
    if a.suggest_renames:
        print(suggest_renames(ev).to_string(index=False)); return
    os.makedirs(a.out, exist_ok=True)
    ev.to_csv(f"{a.out}/broker_evidence.csv", index=False)
    c = Counter(ev.gold_event)
    print(f"wrote {len(ev)} rows for {ev.entity.nunique()} tickers to {a.out}/broker_evidence.csv: {dict(c)}")
    diagnostics(ev)


if __name__ == "__main__":
    main()
