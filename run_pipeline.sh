#!/usr/bin/env bash
# Point-in-time retrieval: the whole experiment in one script.
#
#   setup      create the virtualenv (only if it does not exist yet)            setup_env.sh
#   data       download evidence: ALFRED vintages, SEC EDGAR revenue, Yahoo     data/raw/
#              broker actions (a source is skipped if its file already exists)
#   cap        cap each entity's history so the WHOLE history fits in one       data/real/<source>_evidence.csv
#              prompt (HISTORY_TOKENS, counted with the models' tokenizers)
#   questions  generate questions + gold / peek / trap answers on the capped     data/real/<source>_questions.csv
#              evidence, and the rules cards with real-data examples
#   checks     no-GPU checks: rule engines vs sample gold, downloader -> generator test, dry run of the 4 methods,
#              and method 4's SQL harness with oracle facts (catalogue identifiers must reproduce gold answers)
#   smoke      the 4 methods x every model on a small sample (real models), then check_smoke.py; stops if any fails
#   full       the 4 methods x every model on all questions, then summarize.py                    results/<backend>/
#
# Usage
#   export FRED_API_KEY=...                          # free: https://fredaccount.stlouisfed.org/apikey
#   export SEC_USER_AGENT="Jane Doe jane@uni.edu"    # SEC asks for a name and e-mail
#   nohup bash run_pipeline.sh > run.log 2>&1 &      # everything
#   bash run_pipeline.sh --from smoke                # start at a stage (data already built)
#   bash run_pipeline.sh --only questions            # one stage
#   bash run_pipeline.sh --from full -- --thinking both     # args after -- go to every method script
#
# Settings (environment variables; defaults below). Everything resumes: finished questions, cached fact extraction
# and downloaded files are reused. FORCE_DOWNLOAD=1 re-downloads; OVERWRITE=1 re-runs answered questions.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MODELS="${MODELS:-phi-4-mini qwen3.5-4b gemma-4-e4b-it}"
# data
TICKERS_FILE="${TICKERS_FILE:-$ROOT/data_pipeline/tickers.txt}"
ALFRED_SERIES="${ALFRED_SERIES:-A191RL1Q225SBEA PAYEMS}"
ALFRED_START="${ALFRED_START:-2005-01-01}"
EDGAR_SINCE="${EDGAR_SINCE:-2015-01-01}"
BROKER_SINCE="${BROKER_SINCE:-2016-01-01}"
BROKER_INPUT_CSV="${BROKER_INPUT_CSV:-}"          # optional: existing yfinance notes file (+ ticker column)
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
# history cap and questions
HISTORY_TOKENS="${HISTORY_TOKENS:-12000}"          # evidence tokens per entity (whole history must fit)
ASOF_PER_ENTITY="${ASOF_PER_ENTITY:-4}"
STRATEGY="${STRATEGY:-mixed}"
SEED="${SEED:-7}"
# runs
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-32768}"      # hard prompt ceiling; with the cap nothing should be dropped
SMOKE_LIMIT="${SMOKE_LIMIT:-6}"                    # questions per source in the smoke test
SMOKE_ENTITIES="${SMOKE_ENTITIES:-KO JNJ US_GDP}"  # keeps LLM fact extraction small in the smoke test
OVERWRITE="${OVERWRITE:-0}"

RAW="$ROOT/data/raw"; REAL="$ROOT/data/real"
ALL_STAGES=(setup data cap questions checks smoke full)
FROM=""; ONLY=""; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown argument $1 (use --from STAGE, --only STAGE, -- <method args>)"; exit 2 ;;
  esac
done
run_stage() {   # run_stage <name>: is this stage selected?
  local s="$1"
  if [ -n "$ONLY" ]; then [ "$ONLY" = "$s" ]; return; fi
  if [ -z "$FROM" ]; then return 0; fi
  local hit=0
  for x in "${ALL_STAGES[@]}"; do [ "$x" = "$FROM" ] && hit=1; [ "$x" = "$s" ] && { [ $hit = 1 ]; return; }; done
  return 1
}
for x in $FROM $ONLY; do [[ " ${ALL_STAGES[*]} " == *" $x "* ]] || { echo "unknown stage $x (${ALL_STAGES[*]})"; exit 2; }; done

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
banner() { echo; echo "================================================================ $(date '+%F %T')  $*"; }
hf_id() { case "$1" in
  qwen3.5-4b) echo "Qwen/Qwen3.5-4B" ;; gemma-4-e4b-it) echo "google/gemma-4-E4B-it" ;;
  phi-4-mini) echo "microsoft/Phi-4-mini-instruct" ;; *) echo "" ;; esac; }
echo "log: $LOG"
echo "models: $MODELS | history cap: $HISTORY_TOKENS tokens | max input: $MAX_INPUT_TOKENS | extra method args: ${EXTRA[*]:-none}"

# ------------------------------------------------------------------------------------------------ setup
if run_stage setup; then
  banner "setup"
  # skip only if the venv exists AND its packages import (a half-finished install re-runs setup_env.sh)
  if [ -f "$ROOT/env.sh" ] && bash -c "source '$ROOT/env.sh' && [ -n \"\${VIRTUAL_ENV:-}\" ] && python -c 'import vllm, transformers, pandas, sklearn, requests, yfinance' 2>/dev/null"; then
    echo "virtualenv exists and imports work (env.sh); skipping setup_env.sh"
  else
    bash "$ROOT/setup_env.sh"
  fi
fi
[ -f "$ROOT/env.sh" ] && source "$ROOT/env.sh"
PY="$(command -v python || command -v python3)"
echo "python: $PY"

# ------------------------------------------------------------------------------------------------ data
if run_stage data; then
  banner "data: download evidence -> $RAW"
  mkdir -p "$RAW" "$RAW/api"
  need() { [ "$FORCE_DOWNLOAD" = 1 ] || [ ! -s "$RAW/$1_evidence.csv" ]; }
  if need alfred; then
    : "${FRED_API_KEY:?set FRED_API_KEY (free at https://fredaccount.stlouisfed.org/apikey)}"
    "$PY" data_pipeline/download_alfred.py --series $ALFRED_SERIES --start "$ALFRED_START" --out "$RAW" --raw-dir "$RAW/api"
  else echo "alfred: $RAW/alfred_evidence.csv exists (FORCE_DOWNLOAD=1 to refresh)"; fi
  if need edgar; then
    : "${SEC_USER_AGENT:?set SEC_USER_AGENT, e.g. 'Jane Doe jane@uni.edu'}"
    "$PY" data_pipeline/download_edgar.py --tickers-file "$TICKERS_FILE" --user-agent "$SEC_USER_AGENT" \
      --since "$EDGAR_SINCE" --out "$RAW" --raw-dir "$RAW/api"
  else echo "edgar: $RAW/edgar_evidence.csv exists (FORCE_DOWNLOAD=1 to refresh)"; fi
  if need broker; then
    if [ -n "$BROKER_INPUT_CSV" ]; then
      "$PY" data_pipeline/download_broker.py --tickers-file "$TICKERS_FILE" --input-csv "$BROKER_INPUT_CSV" --since "$BROKER_SINCE" --out "$RAW"
    else
      "$PY" data_pipeline/download_broker.py --tickers-file "$TICKERS_FILE" --since "$BROKER_SINCE" --out "$RAW"
    fi
    echo "broker: check the rename suggestions once:  python data_pipeline/download_broker.py --tickers-file $TICKERS_FILE --suggest-renames"
  else echo "broker: $RAW/broker_evidence.csv exists (FORCE_DOWNLOAD=1 to refresh)"; fi
fi

# ------------------------------------------------------------------------------------------------ cap
if run_stage cap; then
  banner "cap: whole history of every entity <= $HISTORY_TOKENS tokens -> $REAL"
  TOKS=(); for m in $MODELS; do id="$(hf_id "$m")"; [ -n "$id" ] && TOKS+=("$id"); done
  "$PY" data_pipeline/cap_history.py --in-dir "$RAW" --out-dir "$REAL" --max-tokens "$HISTORY_TOKENS" --tokenizers "${TOKS[@]}"
fi

# ------------------------------------------------------------------------------------------------ questions
if run_stage questions; then
  banner "questions: templates, gold / peek / trap answers on the capped evidence"
  "$PY" data_pipeline/generate_questions.py --data-dir "$REAL" --asof-per-entity "$ASOF_PER_ENTITY" --strategy "$STRATEGY" --seed "$SEED"
fi

# ------------------------------------------------------------------------------------------------ checks (no GPU)
if run_stage checks; then
  banner "checks: engines, offline downloader test, dry run of the 4 methods, SQL harness with oracle facts"
  "$PY" data_pipeline/generate_questions.py --selftest
  "$PY" data_pipeline/tests/test_offline.py
  DRY="$ROOT/results_dry"; rm -rf "$DRY" "$ROOT/responses_dry"
  for s in 1_rule_based 2_llm_baseline 3_knowledge_graph 4_code_mcp; do
    "$PY" "methods/$s.py" --dry-run --data-dir "$REAL" --limit 5 --out-dir "$DRY" --responses-dir "$ROOT/responses_dry" \
      --max-input-tokens "$MAX_INPUT_TOKENS" > "$ROOT/logs/dry_$s.log" 2>&1 \
      || { echo "dry run of $s FAILED, see logs/dry_$s.log"; tail -20 "$ROOT/logs/dry_$s.log"; exit 1; }
    echo "dry run ok: $s"
  done
  "$PY" methods/check_sql_oracle.py --data-dir "$REAL"
fi

OVR=(); [ "$OVERWRITE" = 1 ] && OVR=(--overwrite)
run_methods() {   # run_methods <out-dir> <responses-dir> <extra args...>
  local out="$1" resp="$2"; shift 2
  for m in $MODELS; do
    # method 3 before 4: the LLM-extracted fact store is built once per model and reused by method 4
    for s in 1_rule_based 2_llm_baseline 3_knowledge_graph 4_code_mcp; do
      banner "$m  $s  ($out)"
      "$PY" "methods/$s.py" --models "$m" --data-dir "$REAL" --out-dir "$out" --responses-dir "$resp" \
        --max-input-tokens "$MAX_INPUT_TOKENS" "$@" ${EXTRA[@]+"${EXTRA[@]}"}
    done
  done
}

# ------------------------------------------------------------------------------------------------ smoke
if run_stage smoke; then
  banner "smoke: 4 methods x [$MODELS] on $SMOKE_LIMIT questions per source (entities: $SMOKE_ENTITIES)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || { echo "no GPU visible"; exit 1; }
  rm -rf "$ROOT/results_smoke" "$ROOT/responses_smoke"
  run_methods "$ROOT/results_smoke" "$ROOT/responses_smoke" --limit "$SMOKE_LIMIT" --entities $SMOKE_ENTITIES --latency-sample 3
  "$PY" methods/check_smoke.py --out-dir "$ROOT/results_smoke" \
    || { echo "SMOKE TEST FAILED: fix the problems above before the full run (or: bash run_pipeline.sh --from full)"; exit 1; }
fi

# ------------------------------------------------------------------------------------------------ full
if run_stage full; then
  banner "full: 4 methods x [$MODELS] on all questions"
  run_methods "$ROOT/results" "$ROOT/responses" ${OVR[@]+"${OVR[@]}"}
  "$PY" methods/summarize.py --out-dir "$ROOT/results"
  banner "done. Summaries: results/<backend>/summary_all_methods.csv, summary_by_qtype.csv, summary_error_taxonomy.csv"
fi