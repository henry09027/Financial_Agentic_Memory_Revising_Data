#!/usr/bin/env python3
"""
Method 4: code-writing (LLM writes SQL on the spot; run_sql is what an MCP server would expose).

Same fact store as method 3. For every question the HARNESS builds an in-memory SQLite database holding
only the question's entity's facts and evidence learned by the as-of time, so peeking is impossible by
construction. The LLM writes ```sql``` blocks (text tool mode, identical for all three models), gets the
result back, and answers with JSON. --max-calls bounds the queries per question. Questions in a chunk
advance turn by turn in shared batches.

The full conversation (queries + results) is saved in each question's response JSON.

    python 4_code_mcp.py                         # llm + oracle stores (llm store reused from method 3 cache)
    python 4_code_mcp.py --extraction oracle
"""
import json, re, sqlite3, statistics
import pandas as pd
import pit_common as C
from pit_common import S

METHOD = "code_mcp"


def add_args(p):
    g = p.add_argument_group("method 4")
    g.add_argument("--extraction", nargs="+", default=["llm", "oracle"], choices=["llm", "oracle"])
    g.add_argument("--max-calls", type=int, default=6, help="SQL-call budget per question")
    g.add_argument("--max-rows", type=int, default=50, help="rows returned per query")
    g.add_argument("--no-catalogue", action="store_true",
                   help="do not show the as-of database catalogue (exact subject/attribute strings) in the first message")


class Median:
    def __init__(self): self.v = []
    def step(self, x):
        if x is not None:
            try: self.v.append(float(x))
            except (TypeError, ValueError): pass
    def finalize(self): return statistics.median(self.v) if self.v else None


def to_number(x):
    """'65', '$65.00', '1,234.5', '2.3%' -> float (LLM-extracted values are often formatted); text -> None."""
    t = re.sub(r"(?i)\s*(usd|us\$|\$|%|,)\s*", "", str(x).strip()).replace("\u2212", "-")
    try:
        return float(t)
    except ValueError:
        return None


def as_of_db(facts, ev, as_of):
    con = sqlite3.connect(":memory:")
    con.create_aggregate("median", 1, Median)
    cut = pd.Timestamp(as_of)
    f = facts[facts.learned_at <= cut.strftime("%Y-%m-%d %H:%M:%S")].copy()
    f["value_num"] = [to_number(v) for v in f.value]
    f.to_sql("facts", con, index=False)
    e = ev[ev.known_at <= cut][["evidence_id", "entity", "known_at", "text"]].copy()
    e["known_at"] = e.known_at.dt.strftime("%Y-%m-%d %H:%M:%S")
    e.to_sql("evidence", con, index=False)
    con.execute("PRAGMA query_only = ON")
    return con


SQL_BLOCK = re.compile(r"```[ \t]*(?:sqlite3|sqlite|sql)?[ \t]*\n?(.*?)```", re.S | re.I)
BARE_SQL = re.compile(r"(?:^|\n)\s*((?:SELECT|WITH)\b.*?)(?:;|\n\s*\n|$)", re.S | re.I)


def find_sql(text):
    """The query in a reply: a fenced block (```sql, ```sqlite or plain ``` holding a SELECT/WITH), or, when
    the reply has no JSON answer, a bare SELECT/WITH statement. Run 1 ended the conversation on these."""
    for blk in SQL_BLOCK.findall(text or ""):
        if re.match(r"\s*(SELECT|WITH)\b", blk, re.I):
            return blk.strip()
    if re.search(r"\{\s*\"answer\"", text or ""):
        return None
    m = BARE_SQL.search(text or "")
    return m.group(1).strip().rstrip("`").strip() if m else None


def clean_sql(sql):
    """Undo markdown escaping the models copy from the rules card ('AAPL \\| Barclays') and keep one statement."""
    sql = sql.replace("\\|", "|").strip()
    parts = [x for x in re.split(r";\s*(?=(?:[^']*'[^']*')*[^']*$)", sql) if x.strip()]   # ';' outside quotes
    return parts[0] if parts else sql


def run_sql(con, sql):
    n = S.args.max_rows
    sql = clean_sql(sql)
    try:
        cur = con.execute(sql)
        cols = [c[0] for c in cur.description or []]
        rows = cur.fetchmany(n + 1)
    except Exception as e:
        hint = ""
        if "no such column" in str(e) or "no such table" in str(e):
            hint = " Tables and columns: " + SCHEMA_COLS
        return f"SQL error: {e}.{hint}"
    out = pd.DataFrame(rows[:n], columns=cols).to_csv(index=False) + ("...(truncated)" if len(rows) > n else "")
    if not rows:
        return out + "(no rows) " + empty_hint(con, sql)
    return out


def empty_hint(con, sql):
    """What exists close to an empty query: subject strings it did not match exactly, and the stored periods
    sharing a year with the query (smoke test: models asked for '2019-01-01..2019-12-31' when JNJ's fiscal
    year is '2018-12-31..2019-12-29'). Built from the as-of database only, so nothing after the cutoff leaks."""
    hint = ["Check that subject, attribute and period strings are copied exactly (or use LIKE '%...%'), and that "
            "numeric comparisons use value_num."]
    lits = re.findall(r"'((?:[^']|'')*)'", sql)
    subjects = [x for (x,) in con.execute("SELECT DISTINCT subject FROM facts")]
    bad = [x for x in lits if " | " in x and x not in subjects and "%" not in x]
    if bad:
        hint.append("No subject is exactly " + ", ".join(f"'{x}'" for x in bad[:3]) + ".")
    years = sorted(set(re.findall(r"(?<!\d)((?:19|20)\d\d)(?!\d)", " ".join(lits))))
    if years:
        like = " OR ".join(["period LIKE ?"] * len(years))
        per = [x for (x,) in con.execute(f"SELECT DISTINCT period FROM facts WHERE period != '' AND ({like}) "
                                         "ORDER BY period LIMIT 16", [f"%{y}%" for y in years])]
        if per:
            hint.append(f"Stored periods mentioning {', '.join(years)}: " + ", ".join(f"'{x}'" for x in per) + ".")
    return " ".join(hint)


SCHEMA_COLS = ("facts(fact_id, evidence_id, learned_at, subject, attribute, period, value, effective_at, kind, value_num), "
               "evidence(evidence_id, entity, known_at, text)")
SCHEMA = ("Tables: " + SCHEMA_COLS + ". Both contain only what was known at the as-of time, and only "
          "for the entity this question is about. value is text; value_num is the same value as a number (use it for "
          "arithmetic and comparisons). Timestamps are 'YYYY-MM-DD HH:MM:SS' text; effective_at may be empty. "
          "The evidence table holds the original text, searchable with LIKE. "
          "A median(x) aggregate exists. Queries return up to <<N>> rows as CSV.")


def catalogue(con, max_subjects=80):
    """What the as-of database holds, so the model uses the real identifiers (schema linking), plus one
    example query built from them. Run 1 showed no identifiers: models guessed the rules card's sample entity
    (ZETA_GDP) or escaped '|', got empty results and abstained on 40-52% of answerable questions.
    Built from the as-of database itself, so nothing published after the cutoff can appear here."""
    n_f = con.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    n_e = con.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
    if n_f == 0:
        return (f"Database contents at the as-of time: the facts table is empty; the evidence table has {n_e} rows "
                "(query it with LIKE).")
    pairs = con.execute("SELECT subject, attribute, COUNT(*) FROM facts GROUP BY subject, attribute "
                        "ORDER BY subject, attribute").fetchall()
    by_subj = {}
    for sub, att, n in pairs:
        by_subj.setdefault(sub, []).append(f"{att} ({n})")
    subjects = list(by_subj)
    kinds = [k for (k,) in con.execute("SELECT DISTINCT kind FROM facts ORDER BY kind LIMIT 15")]
    periods = [p for (p,) in con.execute("SELECT period FROM facts WHERE period != '' GROUP BY period "
                                         "ORDER BY MAX(learned_at) DESC LIMIT 8")]
    lines = [f"Database contents at the as-of time: {n_f} facts, {n_e} evidence rows.",
             "Exact subject strings (copy them literally; ' | ' is part of the string, never escape it), with their "
             "attributes and row counts" + (f" (first {max_subjects} of {len(subjects)} subjects)" if len(subjects) > max_subjects else "") + ":"]
    lines += [f"  '{sub}': " + ", ".join(by_subj[sub]) for sub in subjects[:max_subjects]]
    lines.append("kind values: " + ", ".join(f"'{k}'" for k in kinds))
    if periods:
        lines.append("most recently learned periods: " + ", ".join(f"'{p}'" for p in periods))
    sub, att, _ = max(pairs, key=lambda x: x[2])
    lines.append(f"Example query: SELECT period, value_num, effective_at, learned_at, kind, evidence_id FROM facts "
                 f"WHERE subject = '{sub.replace(chr(39), chr(39) * 2)}' AND attribute = '{att}' ORDER BY learned_at DESC LIMIT 10")
    return "\n".join(lines)


def code_system(rules):
    return (C.answer_system(rules) + "\nYou can query a read-only SQLite database. " + SCHEMA.replace("<<N>>", str(S.args.max_rows)) +
            "\nTo run a query, reply with exactly one ```sql ... ``` block and nothing else; you will get the result. "
            f"You have up to {S.args.max_calls} queries. When you know the answer, reply with the JSON only.")


def solve(items, thinking):
    """Multi-turn tool loop for a chunk of questions, advanced in lockstep (one batch per turn)."""
    st = [dict(msgs=list(m), con=e.pop("_con"), extra=e, sqls=[], raws=[], tin=0, tout=0, sec=0.0, last=None, done=False,
               n_err=0, n_empty=0)
          for _, _, m, e in items]
    budget = S.args.max_calls
    for step in range(budget + 2):           # after the budget: one "answer now" message, then the final answer
        act = [s for s in st if not s["done"]]
        if not act:
            break
        outs = C.gen([s["msgs"] for s in act], 1000, thinking)
        if S.llm is None and step == 0:      # --dry-run: run the catalogue's example query, so the SQL loop is exercised
            for s, o in zip(act, outs):
                m = re.search(r"Example query: (.*)", s["msgs"][-1]["content"])
                if m:
                    o["text"] = f"```sql\n{m.group(1)}\n```"
        for s, o in zip(act, outs):
            s["last"] = o; s["raws"].append(o["raw"])
            s["tin"] += o["in_tokens"]; s["tout"] += o["out_tokens"]; s["sec"] += o["seconds"]
            sql = find_sql(o["text"])
            if not sql or step == budget + 1:
                s["done"] = True; continue
            s["sqls"].append(sql)
            res = "Budget used up. Answer now with the JSON." if step >= budget else run_sql(s["con"], sql)
            s["n_err"] += res.startswith("SQL error")
            s["n_empty"] += "(no rows)" in res
            s["msgs"] += [{"role": "assistant", "content": o["text"]}, {"role": "user", "content": "Result:\n" + res}]
    out = []
    for s in st:
        s["con"].close()
        res = dict(text=s["last"]["text"] if s["last"] else C.DUMMY, raw="\n\n===== next turn =====\n\n".join(s["raws"]),
                   in_tokens=s["tin"], out_tokens=s["tout"], seconds=s["sec"])
        out.append((res, s["msgs"], dict(s["extra"], n_sql=len(s["sqls"]), n_sql_error=s["n_err"], n_sql_empty=s["n_empty"],
                                       n_turns=len(s["raws"]), sql=json.dumps(s["sqls"]))))
    return out


def smoke_test():
    """No LLM: on the sample data/ the plan's hand-written consensus query must return 55."""
    import os
    d = os.path.join(C.ROOT, "data", "sample")
    if not os.path.exists(f"{d}/broker_evidence.csv"):
        return
    keep = S.args.data_dir, S.args.entities, S.args.limit
    S.args.data_dir, S.args.entities, S.args.limit = d, None, 0
    src = C.Source("broker")
    S.args.data_dir, S.args.entities, S.args.limit = keep
    f, _ = C.build_store(src, "oracle")
    con = as_of_db(C.entity_facts(f, "AUR"), src.ev, "2026-07-28 12:00:00")
    out = run_sql(con, """
WITH t AS (SELECT subject, value_num, effective_at,
           ROW_NUMBER() OVER (PARTITION BY subject ORDER BY effective_at DESC, learned_at DESC) AS rn
           FROM facts WHERE attribute = 'price_target_usd'),
     s AS (SELECT effective_at AS split_at, value_num AS ratio FROM facts WHERE attribute = 'split_ratio')
SELECT median(t.value_num / COALESCE((SELECT ratio FROM s WHERE t.effective_at < s.split_at), 1)) AS consensus
FROM t WHERE rn = 1 AND t.effective_at >= datetime('2026-07-28 12:00:00', '-365 days')""")
    print(f"[harness smoke test on sample data] consensus -> {out.split()[-1]} ({'OK' if '55.0' in out else 'FAILED'})")


def run(sources):
    smoke_test()
    for extraction in S.args.extraction:
        stores, ing = {}, {}
        for name, src in sources.items():
            f, ing[name] = C.build_store(src, extraction)
            stores[name] = {e: C.entity_facts(f, e) for e in src.by_entity}

        def build(src, q):
            ef = stores[src.name].get(q.entity, pd.DataFrame(columns=C.FACT_COLS))
            con = as_of_db(ef, src.by_entity.get(q.entity, src.ev.iloc[:0]), q.as_of)
            user = f"As-of time: {q.as_of}\nQuestion: {q.question}"
            if not S.args.no_catalogue:
                user += "\n\n" + catalogue(con)
            msgs = [{"role": "system", "content": code_system(src.rules)}, {"role": "user", "content": user}]
            return msgs, dict(_con=con, n_dropped=0)

        for t in C.think_values():
            suffix = "_nocat" if S.args.no_catalogue else ""
            C.run_variant(METHOD, f"{extraction}_{C.ttag(t)}{suffix}", sources, build, t, solve=solve, ingestion=ing,
                          settings=dict(extraction=extraction, max_calls=S.args.max_calls, tool_mode="text",
                                        catalogue=not S.args.no_catalogue))


if __name__ == "__main__":
    C.main(METHOD, add_args, run, description=__doc__)