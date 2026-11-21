# Point-in-time retrieval: data pipeline + four methods (run 2)

One project, one script: `run_pipeline.sh` downloads the evidence, caps each entity's history, generates the
questions, checks everything that can be checked without a GPU, runs a small smoke test of the four methods on
every model, and then the full run.

```
pit_pipeline/
  run_pipeline.sh          the whole experiment (stages below)
  setup_env.sh             virtualenv with the pinned vLLM stack (called by run_pipeline.sh if missing)
  requirements.txt, check_env.py
  data_pipeline/
    download_alfred.py     ALFRED vintages (FRED API)              -> data/raw/alfred_evidence.csv
    download_edgar.py      SEC XBRL revenue + acceptance times     -> data/raw/edgar_evidence.csv
    download_broker.py     Yahoo analyst actions + splits          -> data/raw/broker_evidence.csv
    cap_history.py         NEW: whole history per entity <= HISTORY_TOKENS  -> data/real/<source>_evidence.csv
    generate_questions.py  templates + gold / peek / trap answers  -> data/real/<source>_questions.csv, _rules.md
    pit_engines.py         rule engines (one per rules card)
    rules/                 rules cards (templates; real-data examples are filled in per run)
    tickers.txt, broker_renames.csv, tests/
  methods/
    1_rule_based.py  2_llm_baseline.py  3_knowledge_graph.py  4_code_mcp.py
    pit_common.py          shared: engines, data, scoring, audit files, summaries
    summarize.py           combined summaries
    check_sql_oracle.py    NEW: method 4 harness check with oracle facts (no LLM)
    check_smoke.py         NEW: pass/fail check of the smoke run
  data/sample/             the 42-question samples (used by the self-tests)
```

## Run it

```bash
export FRED_API_KEY=...                         # free: https://fredaccount.stlouisfed.org/apikey
export SEC_USER_AGENT="Jane Doe jane@uni.edu"   # SEC asks for a name and e-mail
nohup bash run_pipeline.sh > run.log 2>&1 &     # everything; a copy of the log goes to logs/
```

| stage | what it does | output |
| --- | --- | --- |
| `setup` | creates `WORK/envs/pit` with `setup_env.sh` if `env.sh` has no working virtualenv | `env.sh` |
| `data` | downloads the three sources; a source whose file exists is skipped (`FORCE_DOWNLOAD=1` refreshes) | `data/raw/` |
| `cap` | caps every entity's history to `HISTORY_TOKENS` (see below) | `data/real/*_evidence.csv`, `history_cap.json` |
| `questions` | generates questions on the capped evidence; writes rules cards with real identifiers | `data/real/*_questions.csv`, `*_rules.md` |
| `checks` | engines vs sample gold, downloader→generator test, dry run of all 4 methods, SQL harness with oracle facts | stops on failure |
| `smoke` | 4 methods × every model on `SMOKE_LIMIT` questions per source, then `check_smoke.py` | `results_smoke/`; stops on failure |
| `full` | 4 methods × every model on all questions, then `summarize.py` | `results/<backend>/summary_*.csv` |

```bash
bash run_pipeline.sh --from smoke              # data already built
bash run_pipeline.sh --only questions          # one stage
bash run_pipeline.sh --from full -- --thinking both   # args after -- go to every method script
HISTORY_TOKENS=16000 MODELS="qwen3.5-4b" bash run_pipeline.sh --from cap
```

Settings (environment variables): `MODELS`, `HISTORY_TOKENS` (12000), `MAX_INPUT_TOKENS` (32768), `ASOF_PER_ENTITY` (4),
`STRATEGY` (mixed), `SEED` (7), `SMOKE_LIMIT` (6), `SMOKE_ENTITIES` ("KO JNJ US_GDP"), `TICKERS_FILE`, `ALFRED_SERIES`,
`ALFRED_START`, `EDGAR_SINCE`, `BROKER_SINCE`, `BROKER_INPUT_CSV`, `FORCE_DOWNLOAD`, `OVERWRITE`. Everything resumes:
answered questions, cached fact extraction and downloads are reused.

The smoke test fails a method when it has no rows, any prompt was truncated, fewer than 80% of replies are valid
JSON, more than 10% are empty, or (method 4) no SQL ran or more than 60% of queries errored or returned no rows.
Accuracy on 18 questions is printed but not judged.

## Changes since run 1

### 1. Every entity's whole history fits in the prompt (`cap_history.py`)

Run 1 gave each question its entity's full history, which for big tickers (~35k tokens) and US payrolls (~120k)
exceeded the 32k context: 23% of baseline prompts and 19–26% of knowledge-graph prompts were cut, oldest lines
first. That tests the context limit, not retrieval. Now each entity's history is capped *before* questions are
generated, so:

- every method can see all of an entity's evidence (the baseline sees everything, nothing is dropped);
- gold, peek and trap answers are computed on exactly the evidence the methods see.

The cap keeps the most recent part of each history that fits in `HISTORY_TOKENS`, counted with the models' own
tokenizers (maximum over them; Qwen splits numbers into single digits). It cuts on whole units so no fact is
half-reported: broker notes by time (splits before the window dropped; renames kept when a kept note needs them),
ALFRED by period (a kept period always keeps its real advance estimate and all later revisions), EDGAR by reported
date range. As-of times are sampled after `min(365 days, 40%)` of the capped history. On run 1's downloads with
12k tokens (fallback estimate; the real tokenizers give similar windows): broker windows 1.2–7 years (median 2.7),
payrolls 3.5 years, GDP 8 years, EDGAR 7.5–11.5 years; a baseline prompt is about 13k tokens at most.
`data/real/history_cap.json` records the window of each entity. Raise `HISTORY_TOKENS` for a harder haystack;
the smoke test fails if any prompt gets truncated.

The knowledge-graph prompt is also more compact (the "learned" time and the kind are printed only when they add
information), so with oracle facts it is now smaller than the raw evidence (about 0.4–0.7×).

### 2. Method 4 (text-to-SQL) fixed

Run 1's method 4 gave up on 40–52% of answerable questions. Causes and fixes:

| cause in run 1 | fix |
| --- | --- |
| the model never saw the database's identifiers and copied the rules card's sample entity `ZETA_GDP` | the first message lists the exact subject strings, their attributes and row counts, kind values and recent periods of the **as-of** database, plus an example query built from them (nothing after the cutoff can appear) |
| the rules card's fact schema was a markdown table, so subjects were written `<TICKER> \| <broker>`; models copied the backslash | fact-schema sections rewritten as plain lists with real-data examples (e.g. `TSLA \| Wedbush` is now `TSLA | Wedbush`); the harness also removes `\|` from queries |
| a reply with ```` ```sqlite ````, a plain ```` ``` ```` block or a bare `SELECT` ended the conversation | all of these are run as queries |
| empty results gave no feedback, so the model abstained | an empty result or "no such column" error says what to check |
| `value_num` was empty for LLM-extracted values like `$65.00` | values with `$`, `,`, `%` are parsed |
| multiple statements in one block errored | the first statement is run |

`check_sql_oracle.py` verifies the harness without an LLM: with oracle facts, SQL written only from the catalogue's
identifiers reproduces 100% of the sampled gold answers (ALFRED latest/first print, EDGAR fiscal-year values, broker
consensus with split adjustment, active brokers and missed revisions).

### 3. Knowledge-graph extraction: facts no longer lost (affects methods 3 and 4)

The same `ZETA_GDP` example made LLM extraction write facts under the wrong subject, and `entity_facts()` (subject
must start with the entity) then dropped them. This is the likely reason Gemma's KG-LLM ALFRED accuracy was 0.10 vs
0.64 with oracle facts. Extracted subjects are now attached to the evidence line's entity (`canon_subject`), also
when cached run-1 extractions are reloaded.

### Other

- `json_ok`, `n_sql_error`, `n_sql_empty` per question in the result CSVs.
- A dry run (`--dry-run`) of method 4 now runs the catalogue's example query, so the SQL loop is tested without a GPU.

Run-1 numbers are not comparable with run 2: the evidence, the questions (3,624 on run 1's downloads) and the
prompts changed.

## Outputs

```
results/<backend>/<model>/<method>/<variant>.csv          one row per question (+ _meta.json: settings, cost, wall time)
results/<backend>/summary_all_methods.csv                 accuracy | cost (tokens) | time, per model x method x variant x source
results/<backend>/summary_by_qtype.csv, summary_error_taxonomy.csv
responses/<backend>/<model>/<method>/<variant>/<qid>.json  full prompt (method 4: whole SQL conversation) and raw output
results/<backend>/<model>/cache/                           LLM-extracted facts (shared by methods 3 and 4)
```

## Method flags (pass after `--`)

All four: `--sources`, `--entities`, `--limit N`, `--thinking off|on|both`, `--sample --runs 5`, `--backend`,
`--latency-sample`, `--gpu-mem`, `--prefix-cache`, `--max-input-tokens`, `--load-4bit`, `--overwrite`, `--dry-run`.
Method-specific: `--k --filters` (1), `--modes --orders --no-cheat-sheet` (2), `--extraction llm oracle` (3, 4),
`--max-calls --max-rows --no-catalogue` (4). See `python methods/<script> --help`.

## Environment troubleshooting

| error | fix |
| --- | --- |
| `Disk quota exceeded` | clear `~/.cache/pip`, `~/.cache/huggingface`; `source env.sh` keeps caches under `WORK/.cache` |
| `NVIDIA driver ... too old (found version 12090)` | use the pinned stack (default `setup_env.sh`: vLLM 0.32.0 + CUDA 12.9) |
| `libnvrtc.so.13` (torchcodec) | `pip install --no-deps --force-reinstall "torchcodec==0.16.*" --index-url https://download.pytorch.org/whl/cpu` |
| `GLIBCXX_3.4.30 not found` / `[engine] backend=hf` | the venv is not active: `source env.sh` |
| `ipc path ... longer than 107 characters` | `sed -i '/TMPDIR/d' env.sh && unset TMPDIR` |
| a model asks for a login (Gemma) | `hf auth login` (token stored under `$HF_HOME`) |

## Data notes (unchanged from run 1)

Broker: `GradeDate` treated as UTC → New York; "from $X" appears only on raises/lowers; a "from $X" within 0.5% of
the record is rounding, not a missed revision; renames come from `broker_renames.csv` (check spellings with
`download_broker.py --suggest-renames`); licence: publish the script, not the CSV. ALFRED: first print, the next
two scheduled vintages, then every vintage that changed the value; `known_at` = vintage date 08:30. EDGAR: 10-Q,
10-K and amendments; 3/6/9-month and fiscal-year ranges; `known_at` = acceptance time.
