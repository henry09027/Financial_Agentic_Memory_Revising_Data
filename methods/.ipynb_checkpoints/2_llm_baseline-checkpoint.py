#!/usr/bin/env python3
"""
Method 2: LLM baseline. The LLM gets ALL of the entity's evidence, including notes published after the
as-of time, and must apply the cutoff itself (where peeking happens).

Variants
  all_at_once  x  chrono | shuffled          (default)
  running_note x  chrono | shuffled          (--modes running_note; MemGPT-style memory built once per entity,
                                              one LLM call per evidence line: expensive on real data)
  cheat_sheet                                (only the gold evidence: separates reasoning from retrieval errors)
  x thinking off | on                        (--thinking both; Qwen3.5 and Gemma 4 only)

Long histories: evidence beyond --max-input-tokens is dropped oldest-first and logged per question
(n_dropped); the summary reports truncated_rate.

    python 2_llm_baseline.py
    python 2_llm_baseline.py --models qwen3.5-4b --max-input-tokens 65536
"""
import os, random, time
import pit_common as C
from pit_common import S

METHOD = "llm_baseline"
NOTE_SYSTEM = """You keep a compact running memory of a stream of financial evidence.
Later you will answer questions as of ANY past time, including what was known at that time and how values were revised.
Keep every number with its date/time and evidence id. Rules for this dataset:
<<RULES>>
Return the full updated memory only, at most <<N>> words."""


def add_args(p):
    g = p.add_argument_group("method 2")
    g.add_argument("--modes", nargs="+", default=["all_at_once"], choices=["all_at_once", "running_note"])
    g.add_argument("--orders", nargs="+", default=["chrono", "shuffled"], choices=["chrono", "shuffled"])
    g.add_argument("--no-cheat-sheet", action="store_true")
    g.add_argument("--note-words", type=int, default=400)
    g.add_argument("--seed", type=int, default=7, help="shuffle seed")


def shuffled(rows_or_lines, key):
    x = list(rows_or_lines)
    random.Random(f"{S.args.seed}-{key}").shuffle(x)
    return x


def header_tokens(src, q):
    return C.ntok([C.answer_system(src.rules) + f"As-of time: {q.as_of}\nQuestion: {q.question}\n\nEvidence:\n"])[0] + 64


def build_notes(src, order, thinking):
    """Running note per entity, cached per model; returns ({entity: note}, ingestion cost)."""
    cache = os.path.join(S.args.out_dir, S.model, "cache", f"notes_{src.name}_{order}_{C.ttag(thinking)}")
    os.makedirs(cache, exist_ok=True)
    system = NOTE_SYSTEM.replace("<<RULES>>", src.rules).replace("<<N>>", str(S.args.note_words))
    notes, streams, cost = {}, {}, dict(in_tokens=0, out_tokens=0, seconds=0.0, llm_calls=0)
    for e, g in src.by_entity.items():
        p = f"{cache}/{C._safe(e)}.txt"
        if os.path.exists(p) and S.llm is not None:
            notes[e] = open(p).read()
            for k, v in C.json.load(open(p + ".cost.json")).items():
                cost[k] += v
        else:
            notes[e] = "(empty)"
            rows = list(g.itertuples())
            streams[e] = shuffled(rows, e) if order == "shuffled" else rows
    ecost = {e: dict(in_tokens=0, out_tokens=0, seconds=0.0, llm_calls=0) for e in streams}

    def make_conv(e, r):
        return [{"role": "system", "content": system},
                {"role": "user", "content": f"Current memory:\n{notes[e]}\n\nNew evidence:\n{C.line(r)}"}]

    def update(e, r, o):
        if o["text"] and S.llm is not None:
            notes[e] = o["text"]
        c = ecost[e]
        c["in_tokens"] += o["in_tokens"]; c["out_tokens"] += o["out_tokens"]; c["seconds"] += o["seconds"]; c["llm_calls"] += 1
        if c["llm_calls"] == len(streams[e]) and S.llm is not None:
            open(f"{cache}/{C._safe(e)}.txt", "w").write(notes[e])
            C.json.dump(c, open(f"{cache}/{C._safe(e)}.txt.cost.json", "w"))

    if streams:
        c = C.lockstep(streams, make_conv, update, max_tokens=1200, thinking=thinking, label=f"note {src.name}/{order}")
        for k in cost:
            cost[k] += c[k]
    return notes, cost


def run(sources):
    for t in C.think_values():
        for mode in S.args.modes:
            for order in S.args.orders:
                notes, ing = {}, {}
                if mode == "running_note":
                    for name, src in sources.items():
                        notes[name], ing[name] = build_notes(src, order, t)

                def build(src, q, mode=mode, order=order):
                    if mode == "running_note":
                        body, extra = "Your memory of all evidence:\n" + notes[src.name][q.entity], dict(n_dropped=0)
                    else:
                        g, lines, counts = src.entity_lines(q.entity)
                        kept, dropped = C.fit_lines(lines, S.args.max_input_tokens - header_tokens(src, q), counts)
                        if order == "shuffled":
                            kept = shuffled(kept, q.entity)
                        body, extra = "Evidence:\n" + "\n".join(kept), dict(n_evidence=len(kept), n_dropped=dropped)
                    msgs = [{"role": "system", "content": C.answer_system(src.rules)},
                            {"role": "user", "content": f"As-of time: {q.as_of}\nQuestion: {q.question}\n\n{body}"}]
                    return msgs, extra

                C.run_variant(METHOD, f"{mode}_{order}_{C.ttag(t)}", sources, build, t, ingestion=ing,
                              settings=dict(mode=mode, order=order, note_words=S.args.note_words,
                                            max_input_tokens=S.args.max_input_tokens))
        if not S.args.no_cheat_sheet:
            def build_cheat(src, q):
                ids = set(filter(None, q.gold_evidence.split("|")))
                g = src.by_entity.get(q.entity)
                rows = g[g.evidence_id.isin(ids)] if g is not None else src.ev.iloc[:0]
                body = "Evidence:\n" + "\n".join(C.line(r) for r in rows.itertuples())
                msgs = [{"role": "system", "content": C.answer_system(src.rules)},
                        {"role": "user", "content": f"As-of time: {q.as_of}\nQuestion: {q.question}\n\n{body}"}]
                return msgs, dict(n_evidence=len(rows), n_dropped=0)
            C.run_variant(METHOD, f"cheat_sheet_{C.ttag(t)}", sources, build_cheat, t, settings=dict(mode="cheat_sheet"))


if __name__ == "__main__":
    C.main(METHOD, add_args, run, description=__doc__)
