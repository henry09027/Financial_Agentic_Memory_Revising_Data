#!/usr/bin/env python3
"""No-LLM check of method 4's harness on the generated data.

With the ORACLE fact store, hand-written SQL that uses only the identifiers shown in the as-of catalogue
must reproduce the gold answers of the simple question types. If this fails, method 4 cannot work no
matter how good the model is (run 1's identifier bug would have shown up here).

  python check_sql_oracle.py [--data-dir ../data/real] [--per-type 40]
"""
import argparse, importlib, os, random, re, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pit_common as C          # noqa: E402
from pit_common import S        # noqa: E402
M = importlib.import_module("4_code_mcp")


def catalogue_subjects(text):
    return re.findall(r"^  '(.*)': ", text, re.M)


def sql_for(src, q, subjects):
    """Plain SQL for the simple question types, written only from the catalogue's identifiers."""
    if src == "alfred" and q.qtype in ("latest_value", "old_period_latest", "first_print", "not_yet_published"):
        per = re.search(r"for (\d{4}(?:Q\d|-\d\d))|in (\d{4}(?:Q\d|-\d\d))", q.question)
        if not per:
            return None
        per = per.group(1) or per.group(2)
        order = "ASC" if q.qtype == "first_print" else "DESC"
        return (f"SELECT value_num FROM facts WHERE subject = '{subjects[0]}' AND attribute = 'estimate' "
                f"AND period = '{per}' ORDER BY learned_at {order}, fact_id {order} LIMIT 1")
    if src == "edgar" and q.qtype in ("latest_value", "first_print"):
        m = re.search(r"ended (\d+ \w+ \d{4})", q.question)
        if not m:
            return None
        end = pd.Timestamp(m.group(1)).strftime("%Y-%m-%d")
        order = "ASC" if q.qtype == "first_print" else "DESC"
        return (f"SELECT value_num FROM facts WHERE subject = '{subjects[0]}' AND attribute = 'value_usd_m' "
                f"AND period LIKE '%..{end}' AND julianday(substr(period, 13)) - julianday(substr(period, 1, 10)) > 300 "
                f"ORDER BY learned_at {order}, fact_id {order} LIMIT 1")
    if src == "broker" and q.qtype == "consensus":
        cut = q.as_of
        return f"""
WITH t AS (SELECT subject, value_num, effective_at,
           ROW_NUMBER() OVER (PARTITION BY subject ORDER BY effective_at DESC, learned_at DESC) AS rn
           FROM facts WHERE attribute = 'price_target_usd'),
     act AS (SELECT subject FROM facts WHERE attribute IN ('price_target_usd', 'rating') AND kind != 'inferred'
             GROUP BY subject HAVING MAX(effective_at) >= datetime('{cut}', '-365 days')),
     s AS (SELECT effective_at AS split_at, value_num AS ratio FROM facts WHERE attribute = 'split_ratio')
SELECT ROUND(median(t.value_num / COALESCE((SELECT EXP(SUM(LN(ratio))) FROM s WHERE t.effective_at < s.split_at), 1)), 2)
FROM t WHERE rn = 1 AND t.subject IN (SELECT subject FROM act)"""
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=os.path.join(C.ROOT, "data", "real"))
    p.add_argument("--per-type", type=int, default=40)
    p.add_argument("--min-ok", type=float, default=0.95)
    a = p.parse_args()
    S.args = argparse.Namespace(data_dir=os.path.abspath(a.data_dir), entities=None, limit=0, limit_seed=0,
                                max_rows=50, max_calls=6, sources=["alfred", "edgar", "broker"])
    ok_all, rng = True, random.Random(0)
    for name in ("alfred", "edgar", "broker"):
        if not os.path.exists(f"{S.args.data_dir}/{name}_questions.csv"):
            continue
        src = C.Source(name)
        f, _ = C.build_store(src, "oracle")
        stores = {e: C.entity_facts(f, e) for e in src.by_entity}
        n = good = escaped = nc = cat_ok = 0
        for qtype, g in src.qs.groupby("qtype"):
            for q in rng.sample(list(g.itertuples()), min(a.per_type, len(g))):
                con = M.as_of_db(stores[q.entity], src.by_entity[q.entity], q.as_of)
                cat = M.catalogue(con)
                subj = catalogue_subjects(cat)
                escaped += "\\|" in cat
                if name == "broker" and q.qtype != "consensus":   # catalogue must list the entity's subjects
                    nc += 1; cat_ok += bool(subj) and all(s.startswith(q.entity) for s in subj); con.close(); continue
                sql = sql_for(name, q, subj)
                if sql is None:
                    con.close(); continue
                out = M.run_sql(con, sql).split("\n")
                con.close()
                got = out[1] if len(out) > 1 and out[1] and "no rows" not in out[1] else "CANNOT_ANSWER"
                n += 1
                good += C.match(got, q.gold, q.answer_type, q.tol)
        rate = good / max(1, n)
        ok = (n == 0 or rate >= a.min_ok) and escaped == 0 and cat_ok == nc
        ok_all &= ok
        print(f"[sql-oracle] {name}: {good}/{n} gold answers reproduced from catalogue identifiers ({rate:.0%}); "
              f"catalogue lists the entity's subjects {cat_ok + n}/{nc + n}; escaped '|': {escaped}  {'OK' if ok else 'FAILED'}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()