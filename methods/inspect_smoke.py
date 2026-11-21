#!/usr/bin/env python3
"""Look inside the smoke run (no GPU): is each method failing for the expected reasons, or because of a bug?

  python methods/inspect_smoke.py                       # results_smoke/ and responses_smoke/
  python methods/inspect_smoke.py --examples 5 > smoke_inspect.txt

Prints, per model:
  1. outcome mix per method/variant (correct / peek / trap / false_abstain / wrong)
  2. wrong answers next to the gold answer (spot format or scoring problems)
  3. method 4: what the SQL loop did - queries per question, error and empty-result messages, failing queries
  4. methods 3 + 4: the LLM-extracted fact store - facts per evidence line, attribute names vs. the rules card
"""
import argparse, collections, glob, json, os, re
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--out-dir", default=os.path.join(ROOT, "results_smoke"))
p.add_argument("--responses-dir", default=os.path.join(ROOT, "responses_smoke"))
p.add_argument("--data-dir", default=os.path.join(ROOT, "data", "real"))
p.add_argument("--examples", type=int, default=3)
a = p.parse_args()

SCHEMA_ATTRS = {"broker": {"price_target_usd", "rating", "split_ratio", "alias"},
                "alfred": {"estimate"}, "edgar": {"value_usd_m"}}
CATS = ["correct", "peek", "trap", "false_abstain", "wrong"]
short = lambda x, n=70: re.sub(r"\s+", " ", str(x))[:n]

files = sorted(f for f in glob.glob(f"{a.out_dir}/*/*/*/*.csv") if "/cache/" not in f)
by_model = collections.defaultdict(list)
for f in files:
    by_model[f.split(os.sep)[-3]].append(f)

for model, fs in by_model.items():
    print("=" * 100, f"\n{model}\n" + "=" * 100)
    # 1. outcome mix
    rows = []
    for f in fs:
        d = pd.read_csv(f, keep_default_na=False)
        c = d.category.value_counts(normalize=True)
        rows.append(dict(method=f.split(os.sep)[-2], variant=os.path.basename(f)[:-4], n=len(d),
                         **{k: round(c.get(k, 0), 2) for k in CATS}))
    print("\n1. outcome mix (share of questions)\n" + pd.DataFrame(rows).to_string(index=False))

    # 2. wrong answers vs gold
    print(f"\n2. up to {a.examples} non-correct answers per variant  [qtype | gold | answer | category]")
    for f in fs:
        d = pd.read_csv(f, keep_default_na=False)
        bad = d[d.category != "correct"].head(a.examples)
        if len(bad):
            print(f"  {f.split(os.sep)[-2]}/{os.path.basename(f)[:-4]}")
            for r in bad.itertuples():
                print(f"    {r.source}/{r.qtype:<24} gold={short(r.gold, 30):<30} answer={short(r.answer, 45):<45} {r.category}")

    # 3. SQL loop
    for f in [f for f in fs if f.split(os.sep)[-2] == "code_mcp"]:
        variant = os.path.basename(f)[:-4]
        backend = f.split(os.sep)[-4]
        js = glob.glob(f"{a.responses_dir}/{backend}/{model}/code_mcp/{variant}/*.json") or \
            glob.glob(f"{a.responses_dir}/{model}/code_mcp/{variant}/*.json")
        errs, n_sql, n_q, no_sql, examples = collections.Counter(), 0, 0, 0, []
        for j in js:
            rec = json.load(open(j))
            msgs = rec.get("prompt") or []
            msgs = json.loads(msgs) if isinstance(msgs, str) else msgs
            n_q += 1
            k = 0
            for i, m in enumerate(msgs):
                if m.get("role") == "assistant" and re.search(r"```", m.get("content", "")):
                    k += 1
                if m.get("role") == "user" and i > 1:
                    t = m.get("content", "")
                    if t.startswith("SQL error"):
                        key = re.sub(r"'[^']*'|\"[^\"]*\"|\d+", "_", t.split(".")[0])[:80]
                        errs[key] += 1
                        if len(examples) < a.examples:
                            examples.append((short(msgs[i - 1]["content"], 160), short(t, 120)))
                    elif "(no rows)" in t:
                        errs["(no rows)"] += 1
                        if len(examples) < a.examples:
                            examples.append((short(msgs[i - 1]["content"], 160), "(no rows)"))
            n_sql += k
            no_sql += k == 0
        print(f"\n3. SQL loop, {variant}: {n_q} questions, {n_sql} queries, {no_sql} questions answered without any query")
        for k, v in errs.most_common(8):
            print(f"    {v:>3} x {k}")
        for q_, r_ in examples:
            print(f"    query: {q_}\n      -> {r_}")

    # 4. LLM-extracted facts
    caches = glob.glob(f"{a.out_dir}/*/{model}/cache/facts_*_llm")
    for c in sorted(caches):
        src = re.search(r"facts_(\w+)_llm", c).group(1)
        parts = [pd.read_csv(x, keep_default_na=False, dtype=str) for x in glob.glob(f"{c}/*.csv")]
        if not parts:
            continue
        fx = pd.concat(parts)
        ents = [os.path.basename(x)[:-4] for x in glob.glob(f"{c}/*.csv")]
        lines = 0
        ev_path = f"{a.data_dir}/{src}_evidence.csv"
        if os.path.exists(ev_path):
            ev = pd.read_csv(ev_path, usecols=["entity", "evidence_id"])
            lines = ev[ev.entity.isin(ents)].evidence_id.nunique()
        covered = fx.evidence_id.nunique()
        attrs = fx.attribute.value_counts()
        off = attrs[[x not in SCHEMA_ATTRS.get(src, set()) for x in attrs.index]]
        print(f"\n4. LLM facts, {src} ({', '.join(ents)}): {len(fx)} facts from {covered}/{lines or '?'} evidence lines "
              f"({len(fx) / max(1, covered):.1f} per line); attributes off the rules card: {off.sum()} "
              f"({off.sum() / max(1, len(fx)):.0%})")
        print("    attributes: " + ", ".join(f"{k} ({v})" for k, v in attrs.head(8).items()))
        print("    subjects:   " + ", ".join(f"{k} ({v})" for k, v in fx.subject.value_counts().head(5).items()))
        print("    kinds:      " + ", ".join(f"{k} ({v})" for k, v in fx.kind.value_counts().head(6).items()))
    print()
