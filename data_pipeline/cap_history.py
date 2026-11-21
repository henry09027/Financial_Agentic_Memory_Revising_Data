"""Cap each entity's evidence history so the WHOLE history fits in one LLM prompt.

Why: in run 1 some entities had far more evidence than the context window (US payrolls ~120k tokens, big
tickers ~35k). The baseline then saw only the most recent part of the history (23% of prompts truncated)
and the knowledge-graph prompt was cut too (19-26%). That tests the 32k limit, not retrieval. With the
cap, every method can see all of an entity's evidence, so differences between methods come from how
they select and use evidence.

How: for each entity, keep the most recent part of the history whose rendered evidence lines (exactly as
the methods print them) fit in --max-tokens. The cut is made on whole "units", so no fact is left
half-reported:
  broker  unit = one note (a time window); splits and renames inside the window are kept, splits before
          the window are dropped (no kept target predates them), and a rename line is kept whenever a
          kept note uses the old or new name.
  alfred  unit = one period with ALL its releases (advance ... latest revision); periods are added newest
          first-release first. A kept period therefore always starts with its real first print.
  edgar   unit = one reported date range with all filings reporting it (same logic as alfred).
The world simply starts at the cut. Questions are generated AFTER capping, so gold, peek and trap answers
are computed on exactly the evidence the methods see.

Token counts use the tokenizers of the models in the run (maximum over them) when `transformers` can load
them, otherwise a conservative estimate of 1 token per 3 characters.

  python cap_history.py --in-dir ../data/raw --out-dir ../data/real --max-tokens 12000 \
      --tokenizers Qwen/Qwen3.5-4B google/gemma-4-E4B-it microsoft/Phi-4-mini-instruct
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import pandas as pd

SOURCES = ["broker", "alfred", "edgar"]


# ----------------------------------------------------------------------------- token counting
class Counter_:
    def __init__(self, ids):
        self.toks, self.note = [], []
        for hf_id in ids or []:
            try:
                from transformers import AutoTokenizer
                self.toks.append(AutoTokenizer.from_pretrained(hf_id))
                self.note.append(hf_id)
            except Exception as e:                                      # gated / offline / not installed
                print(f"  [tokenizer] {hf_id} not available ({str(e)[:90]}); skipped", file=sys.stderr)
        if not self.toks:
            self.note = ["estimate: 1 token per digit + 1 per 3 other characters (no tokenizer available)"]

    def count(self, lines):
        if not self.toks:
            # conservative: Qwen-style tokenizers split numbers into single digits
            return [sum(c.isdigit() for c in x) + math.ceil(sum(not c.isdigit() for c in x) / 3.0) + 1 for x in lines]
        best = [0] * len(lines)
        for t in self.toks:
            ids = t(list(lines), add_special_tokens=False).input_ids
            best = [max(b, len(i) + 1) for b, i in zip(best, ids)]      # +1 for the newline
        return best


def render(ev):
    """Same text as pit_common.line() in the methods."""
    t = pd.to_datetime(ev.known_at)
    return [f"[{i}] {k:%Y-%m-%d %H:%M} | {x}" for i, k, x in zip(ev.evidence_id, t, ev.text)]


# ----------------------------------------------------------------------------- capping
def cap_entity(source, g, budget):
    """g: one entity's evidence (with columns t, tok). Returns (kept frame, info dict)."""
    g = g.sort_values(["t", "evidence_id"], kind="stable")
    full_tok = int(g.tok.sum())
    if source == "broker":
        notes = g[g.gold_event == "broker_action"]
        if notes.empty:
            return g.iloc[:0], dict(reason="no broker notes")
        g = g[(g.gold_event != "split") | (g.t >= notes.t.min())]       # splits before coverage: noise
        full_tok = int(g.tok.sum())
        if full_tok <= budget:
            kept = g
        else:
            notes = g[g.gold_event == "broker_action"].iloc[::-1]
            used, t0 = 0, None
            for t, tok in zip(notes.t, notes.tok):
                if used + tok > budget * 0.97:                           # leave room for split / rename lines
                    break
                used += tok; t0 = t
            t0 = t0 if t0 is not None else notes.t.iloc[0]
            # never split one timestamp
            kept_notes = g[(g.gold_event == "broker_action") & (g.t >= t0)]
            names = set(kept_notes.gold_firm.astype(str))
            splits = g[(g.gold_event == "split") & (g.t >= t0)]
            ren = g[g.gold_event == "rename"]
            # a rename inside the window is kept if a kept note uses either name; one before the window only if
            # kept notes use BOTH names (otherwise nothing in the window needs it)
            used = ren.gold_firm.astype(str).isin(names) | ren.gold_new_name.astype(str).isin(names)
            both = ren.gold_firm.astype(str).isin(names) & ren.gold_new_name.astype(str).isin(names)
            ren = ren[(used & (ren.t >= t0)) | both]
            kept = pd.concat([kept_notes, splits, ren]).sort_values(["t", "evidence_id"], kind="stable")
    else:
        key = g.gold_period.astype(str) if source == "alfred" else g.gold_start.astype(str) + ".." + g.gold_end.astype(str)
        g = g.assign(_unit=key.values)
        if full_tok <= budget:
            kept = g
        else:
            first = g.groupby("_unit").t.min().sort_values(ascending=False)
            size = g.groupby("_unit").tok.sum()
            used, units = 0, []
            for u in first.index:
                if used + size[u] > budget:
                    break
                used += size[u]; units.append(u)
            kept = g[g._unit.isin(set(units))]
        kept = kept.drop(columns="_unit")
    info = dict(lines_before=int(len(g)), lines_after=int(len(kept)), tokens_before=full_tok,
                tokens_after=int(kept.tok.sum()), start_before=f"{g.t.min()}", start_after=f"{kept.t.min()}",
                end=f"{g.t.max()}", span_days_after=_span(source, kept))
    return kept, info


def _span(source, kept):
    k = kept[kept.gold_event == "broker_action"] if source == "broker" else kept
    return int((k.t.max() - k.t.min()).days) if len(k) else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", required=True, help="downloaded <source>_evidence.csv (uncapped)")
    p.add_argument("--out-dir", required=True, help="capped <source>_evidence.csv are written here")
    p.add_argument("--sources", nargs="+", default=SOURCES, choices=SOURCES)
    p.add_argument("--max-tokens", type=int, default=12000, help="evidence tokens per entity (whole history)")
    p.add_argument("--tokenizers", nargs="*", default=[], help="Hugging Face ids; the max count over them is used")
    p.add_argument("--min-lines", type=int, default=20, help="drop entities with fewer evidence lines after capping")
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    counter = Counter_(a.tokenizers)
    print(f"[cap] budget {a.max_tokens} evidence tokens per entity; counting with {', '.join(counter.note)}")
    report = dict(max_tokens=a.max_tokens, counted_with=counter.note, sources={})
    for s in a.sources:
        path = f"{a.in_dir}/{s}_evidence.csv"
        if not os.path.exists(path):
            print(f"[cap] {s}: {path} not found, skipped"); continue
        ev = pd.read_csv(path, dtype={"evidence_id": str})
        ev["t"] = pd.to_datetime(ev.known_at)
        ev["tok"] = counter.count(render(ev))
        out, infos = [], {}
        for e, g in ev.groupby("entity", sort=True):
            kept, info = cap_entity(s, g, a.max_tokens)
            if len(kept) < a.min_lines:
                info["dropped_entity"] = True
                print(f"[cap] {s}/{e}: only {len(kept)} lines, entity dropped")
            else:
                out.append(kept)
            infos[e] = info
        res = pd.concat(out).drop(columns=["t", "tok"]) if out else ev.iloc[:0].drop(columns=["t", "tok"])
        res.to_csv(f"{a.out_dir}/{s}_evidence.csv", index=False)
        report["sources"][s] = infos
        tb = [i["tokens_before"] for i in infos.values() if "tokens_before" in i]
        ta = [i["tokens_after"] for i in infos.values() if "tokens_after" in i and not i.get("dropped_entity")]
        capped = sum(1 for i in infos.values() if i.get("lines_after", 0) < i.get("lines_before", 0))
        print(f"[cap] {s}: {len(ev)} -> {len(res)} lines; {capped}/{len(infos)} entities capped; "
              f"tokens per entity max {max(tb or [0])} -> {max(ta or [0])} (mean {sum(ta) / max(1, len(ta)):.0f}); "
              f"shortest kept span {min([i['span_days_after'] for i in infos.values() if 'span_days_after' in i and not i.get('dropped_entity')] or [0])} days")
    json.dump(report, open(f"{a.out_dir}/history_cap.json", "w"), indent=1)
    print(f"[cap] written: {a.out_dir}/<source>_evidence.csv and history_cap.json")


if __name__ == "__main__":
    main()
