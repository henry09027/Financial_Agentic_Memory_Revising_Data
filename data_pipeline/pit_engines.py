"""Point-in-time rule engines: one per benchmark, each a direct translation of its rules card.

Every engine works on the evidence of ONE entity (a ticker or a macro series) and answers the
seven question types at any cutoff timestamp. Each answer comes back as (value, evidence_ids).

Rule switches (e.g. adjust_splits=False, use_inferred=False) deliberately break one rule. The
question generator uses them to compute *trap* answers: the specific wrong answer a method gets
when it forgets that rule. Later cutoffs give the *peek* answers.

Only the gold_* columns and known_at are read here; methods never see these columns.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

CANNOT = "CANNOT_ANSWER"
BUY = {"buy", "strong buy", "outperform", "overweight", "positive", "market outperform"}   # broker rule 4
ONE_SEC = timedelta(seconds=1)


def present(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x)) and str(x) != ""


def fmt(x) -> str:
    """Numbers as plain text: no exponent, no trailing zeros."""
    if isinstance(x, str):
        return x
    s = f"{float(x):.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _prep(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()
    ev["t"] = pd.to_datetime(ev["known_at"])
    return ev.sort_values(["t", "evidence_id"], kind="stable").reset_index(drop=True)


# =============================================================================== BROKER
@dataclass
class Target:
    eff: pd.Timestamp       # when the target became true
    value: float            # split-adjusted (if adjustment is on)
    kind: str               # actual | inferred
    ev: str                 # evidence id that told us
    note_t: pd.Timestamp    # time of the note the stated number is denominated in
    raw: float = None       # number as stated in the note
    base_ev: str = None     # inferred targets: the earlier note the missed revision follows


@dataclass
class BrokerState:
    cut: pd.Timestamp
    targets: dict = field(default_factory=dict)     # canonical firm -> [Target]
    notes: dict = field(default_factory=dict)       # canonical firm -> [row]
    splits: list = field(default_factory=list)      # (time, ratio, evidence_id)
    renames: dict = field(default_factory=dict)     # canonical firm -> [rename evidence ids]
    canon: callable = None

    def target_on(self, firm, t):
        vals = [x for x in self.targets.get(firm, []) if x.eff <= t]
        return max(vals, key=lambda x: x.eff) if vals else None

    def split_ids(self, tgt: Target):
        return [i for s, _, i in self.splits if tgt.note_t < s]

    def active(self, days=365):
        if days is None:
            return list(self.notes)
        lo = self.cut - timedelta(days=days)
        return [f for f, ns in self.notes.items() if ns[-1].t >= lo]


class BrokerEngine:
    # relative gap below which a stated "from $X" is treated as equal to our record (split rounding)
    MISSED_TOL = 0.005

    def __init__(self, ev: pd.DataFrame):
        self.ev = _prep(ev)
        self.entity = str(self.ev.entity.iloc[0])

    def state(self, cut, adjust_splits=True, use_renames=True, use_inferred=True) -> BrokerState:
        cut = pd.Timestamp(cut)
        v = self.ev[self.ev.t <= cut]
        st = BrokerState(cut=cut)
        if adjust_splits:
            st.splits = [(r.t, float(r.gold_split_ratio), r.evidence_id) for r in v.itertuples() if r.gold_event == "split"]
        alias, rename_ev = {}, {}
        if use_renames:
            for r in v[v.gold_event == "rename"].itertuples():
                alias[str(r.gold_new_name)] = str(r.gold_firm)
                rename_ev.setdefault(str(r.gold_new_name), r.evidence_id)

        def canon(f):
            seen = set()
            while f in alias and f not in seen:
                seen.add(f); f = alias[f]
            return f
        st.canon = canon
        for new, eid in rename_ev.items():
            st.renames.setdefault(canon(new), []).append(eid)

        def factor(t):
            p = 1.0
            for s, k, _ in st.splits:
                if t < s:
                    p *= k
            return p

        for r in v[v.gold_event == "broker_action"].itertuples():
            f = canon(str(r.gold_firm))
            st.notes.setdefault(f, []).append(r)
            tl = st.targets.setdefault(f, [])
            if use_inferred and present(r.gold_prior_pt) and tl:                       # rule 6
                prior = float(r.gold_prior_pt) / factor(r.t)
                last = tl[-1].value
                if abs(last - prior) > max(0.01, self.MISSED_TOL * abs(last)):
                    last_actual = [x for x in tl if x.kind == "actual"][-1]
                    tl.append(Target(last_actual.eff + ONE_SEC, prior, "inferred", r.evidence_id, r.t,
                                     float(r.gold_prior_pt), last_actual.ev))
            if present(r.gold_current_pt) and float(r.gold_current_pt) > 0:
                tl.append(Target(r.t, float(r.gold_current_pt) / factor(r.t), "actual", r.evidence_id, r.t,
                                 float(r.gold_current_pt)))
        return st

    # ---- evidence helpers
    @staticmethod
    def _tgt_ids(st, firm, tgt):
        ids = ([tgt.base_ev] if tgt.base_ev else []) + [tgt.ev] + st.split_ids(tgt)
        if any(str(n.gold_firm) != firm for n in st.notes.get(firm, [])):
            ids += st.renames.get(firm, [])
        return ids

    # ---- the seven question types
    def consensus(self, cut, active_days=365, **sw):
        st = self.state(cut, **sw)
        vals, ids = [], []
        for f in st.active(active_days):
            tg = st.target_on(f, st.cut)
            if tg is not None:
                vals.append(tg.value); ids += self._tgt_ids(st, f, tg)
        if not vals:
            return CANNOT, []
        return round(statistics.median(vals), 2), ids

    def latest_action(self, cut, **sw):
        st = self.state(cut, **sw)
        allnotes = [n for ns in st.notes.values() for n in ns]
        if not allnotes:
            return CANNOT, []
        tmax = max(n.t for n in allnotes)
        last = [n for n in allnotes if n.t == tmax]
        if len(last) > 1:
            return None, []                         # ambiguous: several notes at the same second
        r = last[0]
        f = st.canon(str(r.gold_firm))
        tg = st.target_on(f, st.cut)
        if tg is None:
            return None, []
        return f"{r.gold_firm}; {fmt(round(tg.value, 2))}", list(dict.fromkeys([r.evidence_id] + self._tgt_ids(st, f, tg)))

    def change_since_initiation(self, cut, firm, mix_raw_initial=False, **sw):
        st = self.state(cut, **sw)
        f = st.canon(firm)
        acts = [x for x in st.targets.get(f, []) if x.kind == "actual"]
        if not acts:
            return CANNOT, []
        init, now = acts[0], st.target_on(f, st.cut)
        if mix_raw_initial:   # trap: unadjusted initiation target vs split-adjusted current target
            return round((now.value / init.raw - 1) * 100, 2), []
        ids = [init.ev] + st.split_ids(init) + self._tgt_ids(st, f, now)
        return round((now.value / init.value - 1) * 100, 2), list(dict.fromkeys(ids))

    def value_on_date(self, cut, firm, day_end, **sw):
        st = self.state(cut, **sw)
        f = st.canon(firm)
        tg = st.target_on(f, min(pd.Timestamp(day_end), st.cut))
        if tg is None:
            return CANNOT, []
        return round(tg.value, 2), list(dict.fromkeys(self._tgt_ids(st, f, tg)))

    def count_changes(self, cut, start, count_all_notes=False, count_inferred=False, **sw):
        st = self.state(cut, **sw)
        start = pd.Timestamp(start)
        n, ids = 0, []
        for f, ns in st.notes.items():
            for r in ns:
                if r.t < start:
                    continue
                if r.gold_pt_action in ("Raises", "Lowers") or (count_all_notes and present(r.gold_current_pt)):
                    n += 1; ids.append(r.evidence_id)
            if count_inferred:
                n += sum(1 for x in st.targets.get(f, []) if x.kind == "inferred" and x.eff >= start)
        return n, ids

    def count_buy(self, cut, active_days=365, literal_buy=False, **sw):
        st = self.state(cut, **sw)
        n, ids = 0, []
        for f in st.active(active_days):
            g = str(st.notes[f][-1].gold_to_grade).strip().lower()
            if (g == "buy") if literal_buy else (g in BUY):
                n += 1
            ids.append(st.notes[f][-1].evidence_id)                  # every active broker's latest rating
            if any(str(x.gold_firm) != f for x in st.notes[f]):
                ids += st.renames.get(f, [])
        return n, list(dict.fromkeys(ids))

    def event_lookup(self, cut, firm_name, month, **sw):
        """Rating moved to and new (stated) target in a given month, for a broker named as in the question."""
        st = self.state(cut, **sw)
        f = st.canon(firm_name)
        hits = [n for n in st.notes.get(f, []) if f"{n.t:%Y-%m}" == month]
        if not hits:
            return CANNOT, []
        h = hits[-1]
        if not present(h.gold_current_pt):
            return None, []
        ids = [h.evidence_id] + (st.renames.get(f, []) if firm_name != f else [])
        return f"{h.gold_to_grade}; {fmt(float(h.gold_current_pt))}", ids


# =============================================================================== ALFRED
REGULAR = {"advance", "second", "third", "first", "first estimate", "second estimate", "third estimate"}


class AlfredEngine:
    def __init__(self, ev: pd.DataFrame):
        self.ev = _prep(ev)
        self.entity = str(self.ev.entity.iloc[0])

    def releases(self, cut, ignore_annual=False):
        """period -> list of (value, evidence_id) in publication order."""
        v = self.ev[self.ev.t <= pd.Timestamp(cut)]
        if ignore_annual:
            v = v[v.gold_release.astype(str).str.lower().isin(REGULAR)]
        out = {}
        for r in v.itertuples():
            out.setdefault(str(r.gold_period), []).append((float(r.gold_value), r.evidence_id))
        return dict(sorted(out.items()))

    def first_release_time(self):
        return self.ev.groupby("gold_period").t.min().sort_index()

    # pick: which estimate of the period is used -> latest | first | previous
    @staticmethod
    def _pick(rs, which):
        if which == "first":
            return rs[0]
        if which == "previous":
            return rs[-2] if len(rs) > 1 else rs[-1]
        return rs[-1]

    def latest_value(self, cut, period, which="latest", ignore_annual=False):
        rs = self.releases(cut, ignore_annual).get(period)
        if not rs:
            return CANNOT, []
        v, i = self._pick(rs, which)
        return v, [i]

    def first_print(self, cut, period, which="first", ignore_annual=False):
        return self.latest_value(cut, period, which, ignore_annual)

    def total_revision(self, cut, period, which="latest", ignore_annual=False):
        rs = self.releases(cut, ignore_annual).get(period)
        if not rs:
            return CANNOT, []
        last = self._pick(rs, which)
        return round(last[0] - rs[0][0], 4), list(dict.fromkeys([rs[0][1], last[1]]))

    def aggregate(self, cut, which="latest", ignore_annual=False, n=3):
        rel = self.releases(cut, ignore_annual)
        if len(rel) < n:
            return CANNOT, []
        picks = [self._pick(rs, which) for rs in list(rel.values())[-n:]]
        return round(sum(p[0] for p in picks) / n, 4), [p[1] for p in picks]

    def count_revisions(self, cut, period, include_first=False, changes_only=False, ignore_annual=False):
        rs = self.releases(cut, ignore_annual).get(period)
        if not rs:
            return CANNOT, []
        if changes_only:
            n = sum(1 for a, b in zip(rs, rs[1:]) if abs(a[0] - b[0]) > 1e-9)
        else:
            n = len(rs) - (0 if include_first else 1)
        return n, [i for _, i in rs[1:]]

    def old_period_latest(self, cut, period, **kw):
        return self.latest_value(cut, period, **kw)

    def not_yet_published(self, cut, period, **kw):
        return self.latest_value(cut, period, **kw)


# =============================================================================== EDGAR
def dur_class(start, end):
    d = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    if 80 <= d <= 120:
        return "3M"
    if 170 <= d <= 200:
        return "6M"
    if 250 <= d <= 290:
        return "9M"
    if 350 <= d <= 380:
        return "FY"
    return "other"


class EdgarEngine:
    FY_FORMS = {"10-K", "10-K/A"}

    def __init__(self, ev: pd.DataFrame):
        self.ev = _prep(ev)
        self.ev["rng"] = list(zip(self.ev.gold_start.astype(str), self.ev.gold_end.astype(str)))
        self.ev["dc"] = [dur_class(s, e) for s, e in self.ev.rng]
        self.entity = str(self.ev.entity.iloc[0])

    def values(self, cut, use_first=False):
        """(start, end) -> (value, evidence_id): latest (rule 1) or originally reported (rule 2)."""
        v = self.ev[self.ev.t <= pd.Timestamp(cut)]
        g = v.groupby("rng", sort=False)
        pick = g.first() if use_first else g.last()
        return {k: (float(r.gold_value_musd), r.evidence_id) for k, r in pick.iterrows()}

    @staticmethod
    def quarters(vals):
        """(start, end) -> (value, [ids], direct?) for every quarter: direct 3M figures + derived Q4 (rule 4)."""
        q = {k: (v, [i], True) for k, (v, i) in vals.items() if dur_class(*k) == "3M"}
        nine = {k[0]: (k, v, i) for k, (v, i) in vals.items() if dur_class(*k) == "9M"}
        for (s, e), (v, i) in vals.items():
            if dur_class(s, e) != "FY" or s not in nine:
                continue
            (s9, e9), v9, i9 = nine[s]
            if pd.Timestamp(e9) >= pd.Timestamp(e):
                continue
            k = ((pd.Timestamp(e9) + timedelta(days=1)).strftime("%Y-%m-%d"), e)
            if not any(kk[1] == e for kk in q):
                q[k] = (round(v - v9, 3), [i, i9], False)
        return dict(sorted(q.items(), key=lambda kv: kv[0][1]))

    @staticmethod
    def _contiguous(ks):
        return all(1 <= (pd.Timestamp(b[0]) - pd.Timestamp(a[1])).days <= 5 for a, b in zip(ks, ks[1:]))

    def latest_fy(self, cut):
        v = self.ev[(self.ev.t <= pd.Timestamp(cut)) & (self.ev.dc == "FY") & self.ev.gold_form.isin(self.FY_FORMS)]
        return None if v.empty else max(v.rng, key=lambda k: k[1])

    def latest_direct(self, cut, use_first=False):
        qs = self.quarters(self.values(cut, use_first))
        d = [k for k, x in qs.items() if x[2]]
        return (qs, d[-1]) if d else (qs, None)

    # ---- question types
    def latest_value(self, cut, fy, use_first=False):
        x = self.values(cut, use_first).get(tuple(fy))
        return (CANNOT, []) if x is None else (x[0], [x[1]])

    def first_print(self, cut, fy, use_first=True):
        return self.latest_value(cut, fy, use_first)

    def derived_quarter(self, cut, fy, use_first=False, fy_total=False):
        vals = self.values(cut, use_first)
        if fy_total:
            x = vals.get(tuple(fy))
            return (CANNOT, []) if x is None else (x[0], [x[1]])
        for k, (v, ids, direct) in self.quarters(vals).items():
            if k[1] == fy[1]:
                return v, ids
        return CANNOT, []

    def yoy_growth(self, cut, use_first=False, ytd=False):
        qs, k = self.latest_direct(cut, use_first)
        if k is None:
            return CANNOT, []
        end = pd.Timestamp(k[1])
        if ytd:   # trap: year-to-date figure from the same filing vs the year-earlier YTD
            vals = self.values(cut, use_first)
            ytds = [kk for kk in vals if kk[1] == k[1] and dur_class(*kk) in ("6M", "9M")]
            if not ytds:
                return None, []
            cur = ytds[0]
            prev = [kk for kk in vals if dur_class(*kk) == dur_class(*cur)
                    and 358 <= (end - pd.Timestamp(kk[1])).days <= 372]
            if not prev:
                return None, []
            return round((vals[cur][0] / vals[prev[0]][0] - 1) * 100, 3), [vals[cur][1], vals[prev[0]][1]]
        prev = [kk for kk in qs if 358 <= (end - pd.Timestamp(kk[1])).days <= 372]
        if not prev or qs[prev[0]][0] == 0:
            return CANNOT, []
        return round((qs[k][0] / qs[prev[0]][0] - 1) * 100, 3), qs[k][1] + qs[prev[0]][1]

    def count_reports(self, cut, fy, distinct_values=False, original_only=False):
        v = self.ev[(self.ev.t <= pd.Timestamp(cut)) & (self.ev.rng == tuple(fy)) & self.ev.gold_form.isin(self.FY_FORMS)]
        if distinct_values:
            return int(v.gold_value_musd.round(3).nunique()), list(v.evidence_id)
        if original_only:
            v = v[v.gold_form == "10-K"].head(1)
        return int(len(v)), list(v.evidence_id)

    def ttm(self, cut, use_first=False, last_fy=False):
        if last_fy:
            fy = self.latest_fy(cut)
            return self.latest_value(cut, fy) if fy else (None, [])
        qs, k = self.latest_direct(cut, use_first)
        if k is None:
            return CANNOT, []
        ks = [kk for kk in qs if kk[1] <= k[1]][-4:]
        if len(ks) < 4 or not self._contiguous(ks):
            return None, []                              # not answerable cleanly -> template skipped
        return round(sum(qs[kk][0] for kk in ks), 3), [i for kk in ks for i in qs[kk][1]]

    def not_yet_published(self, cut, q, ytd=False):
        vals = self.values(cut)
        if ytd:
            ytds = [kk for kk in vals if kk[1] == q[1] and dur_class(*kk) in ("6M", "9M")]
            return (vals[ytds[0]][0], [vals[ytds[0]][1]]) if ytds else (None, [])
        x = vals.get(tuple(q))
        return (CANNOT, []) if x is None else (x[0], [x[1]])
