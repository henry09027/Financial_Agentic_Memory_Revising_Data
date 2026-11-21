#!/usr/bin/env python3
"""
Method 3: bitemporal knowledge graph (minimal Zep/Graphiti).

Ingestion: an LLM turns each evidence line into facts (subject, attribute, period, value, effective_at, kind),
in time order per entity; learned_at = the line's publication time. --extraction oracle builds perfect facts
from the gold columns instead (upper bound that isolates extraction errors).
Answering: code keeps the entity's facts learned by the as-of time, resolves versions (latest / first /
timeline) and the LLM reads the resolved lines.

The LLM-extracted store is cached per model in results/<model>/cache/ and reused by 4_code_mcp.py.

    python 3_knowledge_graph.py                          # llm + oracle extraction
    python 3_knowledge_graph.py --extraction oracle      # fast: no ingestion LLM calls
"""
import pandas as pd
import pit_common as C
from pit_common import S

METHOD = "knowledge_graph"


def add_args(p):
    g = p.add_argument_group("method 3")
    g.add_argument("--extraction", nargs="+", default=["llm", "oracle"], choices=["llm", "oracle"])


def resolve(f, as_of):
    """The entity's facts learned by as_of, grouped per subject/attribute/period, rendered as compact dated lines.

    Compact format (run 1's verbose lines made the prompt larger than the raw evidence; 19-26% were truncated):
      period facts : <subject> | <attr> | <period>: latest V | first V0 | n reports: V0 [id @learned]; V1 [id @learned]
      timelines    : <subject> | <attr>: V from <effective> [id]; V2 from <effective> [id, learned <t>, inferred] ...
    'learned' is printed only when it differs from the effective time, the kind only when it is not obvious."""
    f = f[f.learned_at <= pd.Timestamp(as_of).strftime("%Y-%m-%d %H:%M:%S")]
    out = []
    for (s, a, p), g in f.groupby(["subject", "attribute", "period"], sort=False):
        g = g.sort_values("learned_at", kind="stable")
        if p:   # one number for a period, possibly reported several times
            vs = "; ".join(f"{x.value} [{x.evidence_id} @{x.learned_at[:16]}]" for x in g.itertuples())
            out.append(f"{s} | {a} | {p}: latest {g.value.iloc[-1]} | first {g.value.iloc[0]} | {len(g)} reports: {vs}")
        else:   # a value that changes over time: timeline by effective time; same effective time -> later-learned wins
            g = g.assign(eff=g.effective_at.where(g.effective_at != "", g.learned_at)).sort_values(["eff", "learned_at"], kind="stable")
            g = g.drop_duplicates("eff", keep="last")
            pts = []
            for x in g.itertuples():
                tags = [x.evidence_id]
                if x.learned_at[:16] != x.eff[:16]:
                    tags.append(f"learned {x.learned_at[:16]}")
                if x.kind not in ("new", "corroborate", "event", ""):
                    tags.append(x.kind)
                pts.append(f"{x.value} from {x.eff[:16]} [{', '.join(tags)}]")
            out.append(f"{s} | {a}: " + "; ".join(pts))
    return out


def run(sources):
    for extraction in S.args.extraction:
        stores, ing = {}, {}
        for name, src in sources.items():
            f, ing[name] = C.build_store(src, extraction)
            stores[name] = {e: C.entity_facts(f, e) for e in src.by_entity}
        head = ("Knowledge graph facts known at the as-of time. Each line: subject | attribute (| period): values with "
                "[evidence id, when learned if different, kind if not new]:\n")

        def build(src, q):
            ef = stores[src.name].get(q.entity, pd.DataFrame(columns=C.FACT_COLS))
            lines = resolve(ef, q.as_of)
            pre = f"As-of time: {q.as_of}\nQuestion: {q.question}\n\n{head}"
            budget = S.args.max_input_tokens - C.ntok([C.answer_system(src.rules) + pre])[0] - 64
            kept, dropped = C.fit_lines(lines, budget)
            msgs = [{"role": "system", "content": C.answer_system(src.rules)}, {"role": "user", "content": pre + "\n".join(kept)}]
            return msgs, dict(n_facts=len(lines), n_dropped=dropped)

        for t in C.think_values():
            C.run_variant(METHOD, f"{extraction}_{C.ttag(t)}", sources, build, t, ingestion=ing,
                          settings=dict(extraction=extraction, max_input_tokens=S.args.max_input_tokens))


if __name__ == "__main__":
    C.main(METHOD, add_args, run, description=__doc__)
