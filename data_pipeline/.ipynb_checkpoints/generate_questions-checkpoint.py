"""Generate point-in-time questions from real evidence, following each benchmark's rules card.

For every entity (ticker / macro series) the script samples as-of times, fills the seven question
templates of its benchmark, and computes in code:
  gold          answer at the cutoff (rules applied exactly)
  peek_answers  answers that use information published after the as-of time
                (end of the same day, the next piece of evidence, and today)
  trap_answers  answers obtained by breaking one rule (stale brokers, unadjusted splits,
                first print instead of latest, year-to-date instead of three-month, ...)
  gold_evidence evidence ids the gold answer rests on

Each question is asked twice, like the samples: mode=today (cutoff = TODAY) and mode=as_of.

Usage
  python generate_questions.py --data-dir ../data/real --sources broker alfred edgar \
      --asof-per-entity 4 --strategy mixed --seed 7
  python generate_questions.py --selftest --sample-dir ../data      # reproduces the sample gold answers
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pit_engines import CANNOT, AlfredEngine, BrokerEngine, EdgarEngine, dur_class, fmt  # noqa: E402

EOD = timedelta(hours=23, minutes=59, seconds=59)
SAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "samples")   # bundled copy of data/
RULES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules")                # rules cards (templates)


def day(t):
    t = pd.Timestamp(t)
    return f"{t.day} {t:%B %Y}"


def poss(name):
    """Possessive: Delta Partners' / Atlas Securities' / Apple Inc.'s"""
    return f"{name}'" if name.endswith("s") else f"{name}'s"


def end_of_day(t):
    return pd.Timestamp(t).normalize() + EOD


# ------------------------------------------------------------------------------- template specs
class Q:
    """One question template instance: text + how to compute gold and traps at any cutoff."""

    def __init__(self, qtype, text, atype, tol, fn, traps=()):
        self.qtype, self.text, self.atype, self.tol, self.fn, self.traps = qtype, text, atype, tol, fn, list(traps)


def broker_templates(eng: BrokerEngine, as_of, today):
    T = eng.entity
    st = eng.state(as_of)
    st_today = eng.state(today)
    out = []
    out.append(Q("consensus", f"What is the consensus (median) price target for {T} across active brokers, in US dollars?",
                 "number", 0.01, lambda c: eng.consensus(c),
                 [lambda c: eng.consensus(c, active_days=None),
                  lambda c: eng.consensus(c, adjust_splits=False),
                  lambda c: eng.consensus(c, use_renames=False),
                  lambda c: eng.consensus(c, active_days=None, adjust_splits=False)]))
    out.append(Q("latest_action", f"Which broker made the most recent rating or price-target action on {T}, and what is that "
                 "broker's price target now (USD)? Answer as 'Broker; target'.", "label", 0,
                 lambda c: eng.latest_action(c),
                 [lambda c: eng.latest_action(c, adjust_splits=False)]))
    # broker whose coverage started with an initiation known at the as-of time (most notes since)
    inits = [f for f, ns in st.notes.items() if str(ns[0].gold_action) == "initiate"
             and any(x.kind == "actual" for x in st.targets.get(f, []))]
    if inits:
        firm = max(inits, key=lambda f: (len(st_today.notes.get(f, [])), f))
        out.append(Q("change_since_initiation", f"By what percent has {poss(firm)} {T} price target changed since it initiated "
                     "coverage? (split-adjusted, in percent)", "number", 0.1,
                     lambda c, f=firm: eng.change_since_initiation(c, f),
                     [lambda c, f=firm: eng.change_since_initiation(c, f, adjust_splits=False),
                      lambda c, f=firm: eng.change_since_initiation(c, f, mix_raw_initial=True)]))
    # value on the day before the as-of day; prefer a broker whose missed revision is revealed later
    d_end = end_of_day(pd.Timestamp(as_of) - timedelta(days=1))
    cands = []
    for f, tl in st_today.targets.items():
        inf_hidden = [x for x in tl if x.kind == "inferred" and x.eff <= d_end
                      and eng.ev.loc[eng.ev.evidence_id == x.ev, "t"].iloc[0] > pd.Timestamp(as_of)]
        has_now = st.target_on(f, d_end) is not None
        if has_now:
            cands.append((len(inf_hidden) > 0, len(st.notes.get(f, [])), f))
    if cands:
        firm = max(cands)[2]
        out.append(Q("value_on_date", f"What was {poss(firm)} {T} price target on {day(d_end)}, in US dollars?", "number", 0.01,
                     lambda c, f=firm: eng.value_on_date(c, f, d_end),
                     [lambda c, f=firm: eng.value_on_date(c, f, d_end, use_inferred=False),
                      lambda c, f=firm: eng.value_on_date(c, f, c)]))
    a = pd.Timestamp(as_of)
    start = (a.replace(day=1) - timedelta(days=1)).replace(day=1)        # first day of the previous month
    out.append(Q("count_changes", f"How many times did brokers raise or lower their {T} price target between {day(start)} "
                 "and the as-of time?", "number", 0,
                 lambda c: eng.count_changes(c, start),
                 [lambda c: eng.count_changes(c, start, count_all_notes=True),
                  lambda c: eng.count_changes(c, start, count_inferred=True),
                  lambda c: eng.count_changes(c, start, count_all_notes=True, count_inferred=True)]))
    out.append(Q("count_buy", f"How many active brokers currently rate {T} Buy or an equivalent rating?", "number", 0,
                 lambda c: eng.count_buy(c),
                 [lambda c: eng.count_buy(c, literal_buy=True),
                  lambda c: eng.count_buy(c, active_days=None),
                  lambda c: eng.count_buy(c, use_renames=False)]))
    # a note published after the as-of time; prefer one under a broker's new name (rename trap)
    later = [n for ns in st_today.notes.values() for n in ns if pd.Timestamp(as_of) < n.t <= pd.Timestamp(today)]
    later = [n for n in later if str(n.gold_to_grade) not in ("", "nan") and pd.notna(n.gold_current_pt)]
    if later:
        renamed = [n for n in later if st_today.canon(str(n.gold_firm)) != str(n.gold_firm)]
        n = (renamed or later)[0]
        month = f"{n.t:%Y-%m}"
        out.append(Q("event_lookup", f"In {n.t:%B %Y}, what rating did {n.gold_firm} move {T} to, and what was its new price "
                     "target (USD)? Answer as 'Rating; target'.", "label", 0,
                     lambda c, f=str(n.gold_firm), m=month: eng.event_lookup(c, f, m), []))
    return out


def alfred_templates(eng: AlfredEngine, as_of, today):
    a, td = pd.Timestamp(as_of), pd.Timestamp(today)
    first = eng.first_release_time()
    pub = [p for p, t in first.items() if t <= a]
    if len(pub) < 3:
        return []
    r0 = eng.ev.iloc[0]
    label = str(r0.get("gold_label", "")) if "gold_label" in eng.ev and pd.notna(r0.get("gold_label")) else None
    unit = str(r0.get("gold_unit", "")) if "gold_unit" in eng.ev and pd.notna(r0.get("gold_unit")) else "percent, annual rate"
    if not label:   # sample style: entity "ZETA_GDP" -> "Zeta's ... real GDP growth"
        label = f"{eng.entity.split('_')[0].title()}'s real GDP growth"
    noun = label if "'s" in label else f"the {label}"
    P, Pprev = pub[-1], pub[-2]
    later = eng.ev[(eng.ev.t > a) & (eng.ev.t <= td)]
    olds = [p for p in pub[-5:-2]]
    revised_later = [p for p in olds if p in set(later.gold_period.astype(str))]
    Pold = (revised_later or olds or [pub[0]])[-1]
    nxt = [p for p, t in first.items() if a < t <= td]
    out = [
        Q("latest_value", f"What is the latest estimate of {noun} for {P} ({unit})?", "number", 0.01,
          lambda c: eng.latest_value(c, P),
          [lambda c: eng.latest_value(c, P, which="first"), lambda c: eng.latest_value(c, P, which="previous")]),
        Q("first_print", f"What was the first published (advance) estimate of {noun} for {Pprev} ({unit})?", "number", 0.01,
          lambda c: eng.first_print(c, Pprev),
          [lambda c: eng.first_print(c, Pprev, which="latest"), lambda c: eng.first_print(c, Pprev, which="previous")]),
        Q("total_revision", f"By how much has the estimate of {noun} for {Pprev} been revised in total since its first "
          f"published estimate (in {unit.split(',')[0]} points)?", "number", 0.01,
          lambda c: eng.total_revision(c, Pprev),
          [lambda c: eng.total_revision(c, Pprev, ignore_annual=True), lambda c: eng.total_revision(c, Pprev, which="previous")]),
        Q("aggregate", f"What is the average of {noun} over the three most recent published periods, using the latest "
          f"estimate for each ({unit})?", "number", 0.015,
          lambda c: eng.aggregate(c),
          [lambda c: eng.aggregate(c, which="first"), lambda c: eng.aggregate(c, which="previous"),
           lambda c: eng.aggregate(c, ignore_annual=True)]),
        Q("count_revisions", f"How many times has the estimate of {noun} for {Pprev} been revised after its first "
          "published estimate?", "number", 0,
          lambda c: eng.count_revisions(c, Pprev),
          [lambda c: eng.count_revisions(c, Pprev, include_first=True), lambda c: eng.count_revisions(c, Pprev, changes_only=True)]),
        Q("old_period_latest", f"What is the latest estimate of {noun} for {Pold} ({unit})?", "number", 0.01,
          lambda c: eng.old_period_latest(c, Pold),
          [lambda c: eng.old_period_latest(c, Pold, ignore_annual=True), lambda c: eng.old_period_latest(c, Pold, which="first")]),
    ]
    if nxt:
        Pn = nxt[0]
        out.append(Q("not_yet_published", f"What was {noun} in {Pn} ({unit})?", "number", 0.01,
                     lambda c: eng.not_yet_published(c, Pn), []))
    return out


def edgar_templates(eng: EdgarEngine, as_of, today):
    a, td = pd.Timestamp(as_of), pd.Timestamp(today)
    r0 = eng.ev.iloc[0]
    name = str(r0.gold_entity_name) if "gold_entity_name" in eng.ev and pd.notna(r0.get("gold_entity_name")) else \
        str(r0.text).split(": ", 1)[1].split(" Revenues")[0].split(" (")[0]
    out = []
    fy = eng.latest_fy(a)
    if fy:
        fyrow = eng.ev[eng.ev.rng == fy].iloc[0]
        fyl = f"fiscal year {int(fyrow.gold_fy)} (ended {day(fy[1])})"
        out += [
            Q("latest_value", f"What is {poss(name)} latest reported total revenue for {fyl}, in $ millions?", "number", 0.5,
              lambda c: eng.latest_value(c, fy), [lambda c: eng.latest_value(c, fy, use_first=True)]),
            Q("first_print", f"What {fyl} total revenue did {name} originally report, in $ millions?", "number", 0.5,
              lambda c: eng.first_print(c, fy), [lambda c: eng.first_print(c, fy, use_first=False)]),
            Q("derived_quarter", f"What was {poss(name)} revenue for the three months ended {day(fy[1])} (the fourth quarter of "
              f"fiscal year {int(fyrow.gold_fy)}), in $ millions?", "number", 0.5,
              lambda c: eng.derived_quarter(c, fy),
              [lambda c: (CANNOT, []), lambda c: eng.derived_quarter(c, fy, use_first=True),
               lambda c: eng.derived_quarter(c, fy, fy_total=True)]),
            Q("count_reports", f"How many filings have reported a {fyl} total revenue figure for {name}?", "number", 0,
              lambda c: eng.count_reports(c, fy),
              [lambda c: eng.count_reports(c, fy, original_only=True), lambda c: eng.count_reports(c, fy, distinct_values=True)]),
        ]
        out.sort(key=lambda q: ["latest_value", "first_print", "derived_quarter", "count_reports"].index(q.qtype))
    out.insert(min(3, len(out)), Q("yoy_growth", f"What was the year-over-year revenue growth (percent) of {poss(name)} most recently "
                                   "reported fiscal quarter (three-month figures)?", "number", 0.05,
                                   lambda c: eng.yoy_growth(c),
                                   [lambda c: eng.yoy_growth(c, ytd=True), lambda c: eng.yoy_growth(c, use_first=True)]))
    out.append(Q("ttm", f"What is {poss(name)} trailing-twelve-month revenue through its most recently reported quarter, in $ millions?",
                 "number", 0.5, lambda c: eng.ttm(c),
                 [lambda c: eng.ttm(c, use_first=True), lambda c: eng.ttm(c, last_fy=True)]))
    _, k = eng.latest_direct(a)
    if k is not None:
        nxt = eng.ev[(eng.ev.t > a) & (eng.ev.t <= td) & (eng.ev.dc == "3M")
                     & (eng.ev.gold_end.astype(str) > k[1]) & eng.ev.gold_form.str.startswith("10-Q")]
        if not nxt.empty:
            q = nxt.sort_values("gold_end").rng.iloc[0]
            out.append(Q("not_yet_published", f"What was {poss(name)} revenue for the three months ended {day(q[1])}, in $ millions?",
                         "number", 0.5, lambda c, q=q: eng.not_yet_published(c, q),
                         [lambda c, q=q: eng.not_yet_published(td, q, ytd=True)]))   # YTD from that future filing
    return out


SOURCES = {"broker": (BrokerEngine, broker_templates), "alfred": (AlfredEngine, alfred_templates),
           "edgar": (EdgarEngine, edgar_templates)}


# ------------------------------------------------------------------------------- answers -> rows
def same(a, b, tol):
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    return abs(float(a) - float(b)) <= max(float(tol), 1e-9)


def collect(vals, gold, tol, exclude=()):
    out = []
    for v in vals:
        if v is None or same(v, gold, tol) or any(same(v, x, tol) for x in list(exclude) + out):
            continue
        if v == CANNOT and gold == CANNOT:
            continue
        out.append(v)
    return out


def future_cutoffs(eng, as_of, today, n_next=2):
    """Cutoffs that leak the future: end of the as-of day, the next few evidence times, today."""
    a = pd.Timestamp(as_of)
    nxt = list(dict.fromkeys(eng.ev[eng.ev.t > a].t))[:n_next]
    cs = [end_of_day(a)] + nxt + [pd.Timestamp(today)]
    return [c for c in cs if a < c <= pd.Timestamp(today)]


def rows_for(source, eng, q, k, i, as_of, today):
    rows = []
    for mode, cut in (("today", pd.Timestamp(today)), ("as_of", pd.Timestamp(as_of))):
        gold, gev = q.fn(cut)
        if gold is None:
            return []          # template not answerable unambiguously -> drop both modes
        peeks = [] if mode == "today" else collect([q.fn(c)[0] for c in future_cutoffs(eng, cut, today)], gold, q.tol)
        traps = collect([t(cut)[0] for t in q.traps], gold, q.tol, exclude=peeks)
        rows.append(dict(qid=f"{source}-{eng.entity}-{k:03d}-Q{i}-{mode}", source=source, entity=eng.entity, qtype=q.qtype,
                         mode=mode, as_of=f"{cut:%Y-%m-%d %H:%M:%S}", question=q.text, answer_type=q.atype, tol=q.tol,
                         gold=fmt(gold), peek_answers="|".join(fmt(x) for x in peeks),
                         trap_answers="|".join(fmt(x) for x in traps), gold_evidence="|".join(dict.fromkeys(gev))))
    return rows


# ------------------------------------------------------------------------------- as-of sampling
def event_times(source, eng):
    ev = eng.ev
    if source == "broker":
        sel = ev[(ev.gold_event != "broker_action") | ev.gold_prior_pt.notna()]
    elif source == "alfred":
        sel = ev[~ev.gold_release.astype(str).str.lower().isin({"advance", "first", "first estimate"})]
    else:
        sel = ev[ev.gold_form.astype(str).str.endswith("/A") | ev.text.str.contains("comparative")]
    return list(sel.t)


def sample_asofs(source, eng, n, strategy, today, rng, min_history_days=365, min_history_frac=0.4):
    # earliest as-of: first evidence + min_history_days, but at most min_history_frac of the (capped) history,
    # so entities whose history was capped to a short window still get questions
    ev = eng.ev[eng.ev.gold_event == "broker_action"] if source == "broker" else eng.ev
    start = ev.t.min()                       # broker: first note (an old rename or split line is not history)
    span = pd.Timestamp(today) - start
    lo = start + min(timedelta(days=min_history_days), span * min_history_frac)
    hi = pd.Timestamp(today) - timedelta(days=7)
    if lo >= hi:
        return []
    out = []
    evs = [t for t in event_times(source, eng) if lo < t < hi]
    n_ev = n if strategy == "events" else (0 if strategy == "random" else (n + 1) // 2)
    for t in rng.sample(evs, min(n_ev, len(evs))):              # just before a revision/split/rename/restatement
        out.append(t - timedelta(minutes=rng.randint(60, 24 * 60)))
    while len(out) < n:                                          # uniform over the history
        out.append(lo + (hi - lo) * rng.random())
    return sorted(pd.Timestamp(t).floor("min") for t in out)


def generate(source, ev, today, n_asof, strategy, seed, entities=None, min_history_days=365, min_history_frac=0.4):
    Eng, templates = SOURCES[source]
    rng = random.Random(seed)
    rows, skipped = [], {}
    for ent, g in ev.groupby("entity", sort=True):
        if entities and ent not in entities:
            continue
        eng = Eng(g)
        seen_today = set()
        for k, as_of in enumerate(sample_asofs(source, eng, n_asof, strategy, today, rng, min_history_days, min_history_frac), 1):
            for i, q in enumerate(templates(eng, as_of, today), 1):
                rs = rows_for(source, eng, q, k, i, as_of, today)
                if not rs:
                    skipped[q.qtype] = skipped.get(q.qtype, 0) + 1
                    continue
                if (q.text, rs[0]["gold"]) in seen_today:            # same today-question already asked
                    rs = rs[1:]
                seen_today.add((q.text, rs[0]["gold"] if rs[0]["mode"] == "today" else None))
                rows += rs
    return pd.DataFrame(rows), skipped


# ------------------------------------------------------------------------------- self-test on the samples
SAMPLE_TODAY = "2026-11-14 23:59:59"
SAMPLE_ASOF = {"broker": "2026-07-28 12:00:00", "alfred": "2026-08-15 12:00:00", "edgar": "2026-06-01 12:00:00"}


def selftest(sample_dir):
    ok = True
    for src, (Eng, templates) in SOURCES.items():
        ev = pd.read_csv(f"{sample_dir}/{src}_evidence.csv")
        ref = pd.read_csv(f"{sample_dir}/{src}_questions.csv", keep_default_na=False, dtype=str)
        eng = Eng(ev)
        got = {}
        for i, q in enumerate(templates(eng, SAMPLE_ASOF[src], SAMPLE_TODAY), 1):
            for r in rows_for(src, eng, q, 1, i, SAMPLE_ASOF[src], SAMPLE_TODAY):
                got[(r["qtype"], r["mode"])] = r
        n_ok = n_ev = 0
        for r in ref.itertuples():
            g = got.get((r.qtype, r.mode))
            match = g is not None and same(g["gold"] if g["gold"] == CANNOT else float(g["gold"]) if r.answer_type == "number"
                                           else g["gold"], r.gold if r.gold == CANNOT or r.answer_type != "number"
                                           else float(r.gold), float(r.tol))
            n_ok += match
            n_ev += g is not None and set(g["gold_evidence"].split("|")) == set(r.gold_evidence.split("|"))
            if not match:
                ok = False
                print(f"  MISMATCH {src} {r.qtype} {r.mode}: expected {r.gold}, got {None if g is None else g['gold']}")
        print(f"{src}: gold {n_ok}/{len(ref)} match the sample; gold_evidence identical for {n_ev}/{len(ref)}")
    return ok


# ------------------------------------------------------------------------------- CLI
RULE_NOTES = {
    "broker": "\nNote for this dataset: a stated \"from $X\" within 0.5% of the broker's last recorded target "
              "(rounding after a split) is not a missed revision.\n",
}


def rule_examples(source, ev):
    """Real identifiers of this dataset for the rules card's fact-schema examples (so no method copies a
    sample entity such as ZETA_GDP, which matched nothing in the real data in run 1)."""
    ev = ev.copy()
    ent = ev.entity.value_counts().index[0]
    g = ev[ev.entity == ent]
    if source == "broker":
        firm = g[g.gold_event == "broker_action"].gold_firm.astype(str).value_counts().index[0]
        return dict(ENTITY_EX=ent, SUBJECT_EX=f"{ent} | {firm}")
    if source == "alfred":
        label = str(g.gold_label.iloc[0]) if "gold_label" in g and pd.notna(g.gold_label.iloc[0]) else "real GDP growth % SAAR"
        return dict(ENTITY_EX=ent, SUBJECT_EX=f"{ent} | {label}", PERIOD_EX=str(g.gold_period.iloc[-1]))
    r = g.iloc[-1]
    return dict(ENTITY_EX=ent, SUBJECT_EX=f"{ent} | {r.gold_concept}", PERIOD_EX=f"{r.gold_start}..{r.gold_end}")


def write_rules(source, ev, rules_dir, out_path):
    """Copy the rules card next to the questions, with real-data examples filled in and dataset notes appended."""
    src = f"{rules_dir}/{source}_rules.md"
    if not os.path.exists(src):
        print(f"{source}: rules card {src} not found"); return
    txt = open(src).read()
    for k, v in rule_examples(source, ev).items():
        txt = txt.replace("{" + k + "}", v)
    txt = txt.rstrip("\n") + "\n" + RULE_NOTES.get(source, "")
    with open(out_path, "w") as fh:
        fh.write(txt)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "real"), help="folder with <source>_evidence.csv; questions are written here")
    p.add_argument("--rules-dir", default=RULES,
                   help="folder with the <source>_rules.md cards (copied next to the generated questions)")
    p.add_argument("--sources", nargs="+", default=["broker", "alfred", "edgar"], choices=list(SOURCES))
    p.add_argument("--today", default=None, help="'today' cutoff, e.g. '2026-11-14 23:59:59' (default: end of the day of the latest evidence)")
    p.add_argument("--asof-per-entity", type=int, default=4)
    p.add_argument("--strategy", default="mixed", choices=["events", "random", "mixed"],
                   help="events = shortly before revisions/splits/renames/restatements; random = uniform; mixed = half each")
    p.add_argument("--min-history-days", type=int, default=365, help="earliest as-of time = first evidence + this ...")
    p.add_argument("--min-history-frac", type=float, default=0.4, help="... but at most this share of the entity's history")
    p.add_argument("--entities", nargs="*", default=None, help="restrict to these entities")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--selftest", action="store_true", help="check the engines against the sample gold answers and exit")
    p.add_argument("--sample-dir", default=SAMPLES, help="sample data (bundled copy in tests/samples)")
    a = p.parse_args()

    if a.selftest:
        sys.exit(0 if selftest(a.sample_dir) else 1)

    for src in a.sources:
        path = f"{a.data_dir}/{src}_evidence.csv"
        if not os.path.exists(path):
            print(f"{src}: {path} not found, skipped"); continue
        ev = pd.read_csv(path)
        today = a.today or f"{pd.to_datetime(ev.known_at).max():%Y-%m-%d} 23:59:59"
        qs, skipped = generate(src, ev, today, a.asof_per_entity, a.strategy, a.seed, a.entities, a.min_history_days,
                               a.min_history_frac)
        qs.to_csv(f"{a.data_dir}/{src}_questions.csv", index=False)
        write_rules(src, ev, a.rules_dir, f"{a.data_dir}/{src}_rules.md")
        if len(qs):
            asof = qs[qs["mode"] == "as_of"]
            diff = (asof.set_index(["entity", "question"]).gold != qs[qs["mode"] == "today"].drop_duplicates(["entity", "question"])
                    .set_index(["entity", "question"]).gold.reindex(asof.set_index(["entity", "question"]).index)).mean()
            print(f"{src}: {len(qs)} questions for {qs.entity.nunique()} entities (today={today}); "
                  f"{diff:.0%} of as-of golds differ from today's; with traps {(qs.trap_answers != '').mean():.0%}, "
                  f"with peeks {(asof.peek_answers != '').mean():.0%} of as-of; skipped templates {skipped or 'none'}")
        else:
            print(f"{src}: no questions generated (history too short?)")


if __name__ == "__main__":
    main()
