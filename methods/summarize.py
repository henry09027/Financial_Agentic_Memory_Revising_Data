#!/usr/bin/env python3
"""
Combined summary across the four methods, per inference engine (results/<backend>/ ...).
Timing is only comparable within one engine, so engines are summarised separately.

  results/<backend>/summary_all_methods.csv     accuracy | cost (tokens) | time (s) per model x method x variant x source
  results/<backend>/summary_by_qtype.csv        accuracy per question type x mode
  results/<backend>/summary_error_taxonomy.csv  correct / peek / trap / false_abstain / wrong shares

    python summarize.py [--out-dir results] [--backend vllm]
"""
import argparse, glob, os
import pandas as pd
import pit_common as C

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--out-dir", default=os.path.join(C.ROOT, "results"))
p.add_argument("--backend", default=None, help="only this engine folder (vllm, hf, dry)")
a = p.parse_args()

backends = [a.backend] if a.backend else sorted(
    b for b in os.listdir(a.out_dir) if glob.glob(f"{a.out_dir}/{b}/*/*/*_meta.json"))
for b in backends:
    d = os.path.join(a.out_dir, b)
    print(f"\n######## engine: {b}")
    C.summarize(d)
    res, _ = C._load_results(d)
    if res is None:
        continue
    res["variant"] = res.variant.str.replace(r"_r\d+$", "", regex=True)
    res.pivot_table(index=["source", "qtype", "mode"], columns=["model", "method", "variant"], values="correct",
                    aggfunc="mean").round(3).to_csv(f"{d}/summary_by_qtype.csv")
    pd.crosstab([res.model, res.method, res.variant, res.source], res.category, normalize="index").round(4) \
      .to_csv(f"{d}/summary_error_taxonomy.csv")
    print(f"written: {d}/summary_by_qtype.csv, summary_error_taxonomy.csv")
