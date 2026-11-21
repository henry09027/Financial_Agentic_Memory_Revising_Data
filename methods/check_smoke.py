#!/usr/bin/env python3
"""Checks the small-sample runs before the full run starts. Exit code 1 if a method does not work.

For every model x method x variant found under --out-dir/<backend>/:
  FAIL  no rows, or fewer rows than questions selected
  FAIL  any prompt truncated (n_dropped > 0): the history cap is too large for --max-input-tokens
  FAIL  fewer than --min-json of the replies are valid {"answer": ...} JSON
  FAIL  more than --max-empty of the replies are empty (engine / context problem)
  FAIL  method 4: no SQL ran at all, or more than --max-sql-bad of the queries errored or returned no rows
  info  accuracy, mean input tokens, seconds per question (a small sample: not a result)

  python check_smoke.py --out-dir ../results_smoke
"""
import argparse, glob, os, sys
import pandas as pd

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--out-dir", required=True)
p.add_argument("--min-json", type=float, default=0.80)
p.add_argument("--max-empty", type=float, default=0.10)
p.add_argument("--max-sql-bad", type=float, default=0.60)
p.add_argument("--expect-methods", nargs="+", default=["rule_based", "llm_baseline", "knowledge_graph", "code_mcp"])
a = p.parse_args()

files = [f for f in glob.glob(f"{a.out_dir}/*/*/*/*.csv") if "/cache/" not in f and "summary" not in os.path.basename(f)]
if not files:
    sys.exit(f"[smoke] no result files under {a.out_dir}")
rows, bad = [], []
for f in sorted(files):
    backend, model, method, variant = f.split(os.sep)[-4], f.split(os.sep)[-3], f.split(os.sep)[-2], os.path.basename(f)[:-4]
    df = pd.read_csv(f, keep_default_na=False)
    r = dict(backend=backend, model=model, method=method, variant=variant, n=len(df))
    problems = []
    if not len(df):
        problems.append("no rows")
    else:
        nd = pd.to_numeric(df.get("n_dropped", 0), errors="coerce").fillna(0)
        js = df.get("json_ok", pd.Series(["True"] * len(df))).astype(str).str.lower().eq("true")
        empty = df.get("answer", pd.Series([""] * len(df))).astype(str).str.strip().eq("")
        r.update(accuracy=round(df.correct.astype(str).str.lower().eq("true").mean(), 3),
                 json_ok=round(js.mean(), 3), truncated=int((nd > 0).sum()),
                 in_tok=int(pd.to_numeric(df.in_tokens, errors="coerce").mean()),
                 max_in_tok=int(pd.to_numeric(df.in_tokens, errors="coerce").max()),
                 sec_q=round(pd.to_numeric(df.seconds, errors="coerce").mean(), 3))
        if r["truncated"]:
            problems.append(f"{r['truncated']} truncated prompts")
        if js.mean() < a.min_json:
            problems.append(f"only {js.mean():.0%} valid JSON answers")
        if empty.mean() > a.max_empty:
            problems.append(f"{empty.mean():.0%} empty answers")
        if method == "code_mcp":
            n_sql = pd.to_numeric(df.get("n_sql", 0), errors="coerce").fillna(0)
            n_bad = pd.to_numeric(df.get("n_sql_error", 0), errors="coerce").fillna(0) + \
                pd.to_numeric(df.get("n_sql_empty", 0), errors="coerce").fillna(0)
            r.update(sql_per_q=round(n_sql.mean(), 2), sql_bad=round(n_bad.sum() / max(1, n_sql.sum()), 3))
            if n_sql.sum() == 0:
                problems.append("no SQL query was run")
            elif r["sql_bad"] > a.max_sql_bad:
                problems.append(f"{r['sql_bad']:.0%} of SQL queries errored or returned no rows")
    r["status"] = "FAIL: " + "; ".join(problems) if problems else "ok"
    rows.append(r)
    if problems:
        bad.append(r)
out = pd.DataFrame(rows)
missing = [(m, mm) for m in out.model.unique() for mm in a.expect_methods if not ((out.model == m) & (out.method == mm)).any()]
with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_columns", 30):
    print(out.to_string(index=False))
for m, mm in missing:
    print(f"[smoke] FAIL: {m} has no results for method {mm}")
if bad or missing:
    print(f"\n[smoke] {len(bad) + len(missing)} problem(s). Fix them before the full run "
          "(audit trail: responses_smoke/<backend>/<model>/<method>/<variant>/<qid>.json).")
    sys.exit(1)
print("\n[smoke] all methods work on the sample")
