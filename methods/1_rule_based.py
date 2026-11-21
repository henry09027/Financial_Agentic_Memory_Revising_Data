#!/usr/bin/env python3
"""
Method 1: rule-based retrieval (Generative Agents style).

For each question: take the question's entity's evidence, drop notes published after the as-of time
(--filters timestamp | day | none), score each note by TF-IDF similarity + recency + importance, and pass
the top-k to the LLM.

    python 1_rule_based.py                              # all 3 models, data/real/
    python 1_rule_based.py --models phi-4-mini --limit 20
    python 1_rule_based.py --filters timestamp day      # same-day-leak variant too

Outputs: responses/<model>/rule_based/<variant>/<qid>.json, results/<model>/rule_based/<variant>.csv,
         results/summary_rule_based.csv
"""
import time
import numpy as np, pandas as pd
import pit_common as C
from pit_common import S

METHOD = "rule_based"
IMPORTANT_WORDS = ["revis", "restat", "amend", "correct", "split", "name", "upgrade", "downgrade",
                   "initiat", "advance estimate", "first estimate", "annual update", "benchmark", "10-k"]


def add_args(p):
    g = p.add_argument_group("method 1")
    g.add_argument("--k", type=int, default=3, help="notes passed to the LLM")
    g.add_argument("--filters", nargs="+", default=["timestamp"], choices=["timestamp", "day", "none"])
    g.add_argument("--w-sim", type=float, default=1.0)
    g.add_argument("--w-rec", type=float, default=1.0)
    g.add_argument("--w-imp", type=float, default=0.5)
    g.add_argument("--tau-days", type=float, default=90)


def index(src):
    """TF-IDF over the source's evidence (no LLM). Fit once per source."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    t = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2)).fit(src.ev.text)
    imp = src.ev.text.str.lower().apply(lambda x: min(1.0, sum(w in x for w in IMPORTANT_WORDS) / 2)).values
    pos = {e: np.where(src.ev.entity.values == e)[0] for e in src.by_entity}
    return dict(vec=vec, X=vec.transform(src.ev.text), imp=imp, pos=pos), time.time() - t


def retrieve(src, st, q, flt):
    from sklearn.metrics.pairwise import cosine_similarity
    a, ev, as_of = S.args, src.ev, pd.Timestamp(q.as_of)
    idx = st["pos"].get(q.entity, np.array([], dtype=int))
    t = ev.known_at.values[idx]
    if flt == "timestamp": idx = idx[t <= as_of.to_datetime64()]
    elif flt == "day":     idx = idx[t.astype("datetime64[D]") <= as_of.normalize().to_datetime64()]
    if len(idx) == 0:
        return ev.iloc[:0]
    sim = cosine_similarity(st["vec"].transform([q.question]), st["X"][idx]).ravel()
    sim = sim / sim.max() if sim.max() > 0 else sim
    age = np.clip((as_of - ev.known_at.iloc[idx]).dt.total_seconds().values / 86400, 0, None)
    s = a.w_sim * sim + a.w_rec * np.exp(-age / a.tau_days) + a.w_imp * st["imp"][idx]
    return ev.iloc[idx[np.argsort(-s)[:a.k]]].sort_values("known_at")


def run(sources):
    states, ing = {}, {}
    for name, src in sources.items():
        states[name], sec = index(src)
        ing[name] = dict(in_tokens=0, out_tokens=0, seconds=round(sec, 3), llm_calls=0)
    for flt in S.args.filters:
        def build(src, q, flt=flt):
            top = retrieve(src, states[src.name], q, flt)
            prompt = f"As-of time: {q.as_of}\nQuestion: {q.question}\n\nEvidence:\n" + "\n".join(C.line(r) for r in top.itertuples())
            msgs = [{"role": "system", "content": C.answer_system(src.rules)}, {"role": "user", "content": prompt}]
            return msgs, dict(retrieved="|".join(top.evidence_id), n_dropped=0)
        for t in C.think_values():
            C.run_variant(METHOD, f"k{S.args.k}_{flt}_{C.ttag(t)}", sources, build, t, ingestion=ing,
                          settings=dict(k=S.args.k, filter=flt, w_sim=S.args.w_sim, w_rec=S.args.w_rec,
                                        w_imp=S.args.w_imp, tau_days=S.args.tau_days))


if __name__ == "__main__":
    C.main(METHOD, add_args, run, description=__doc__)
