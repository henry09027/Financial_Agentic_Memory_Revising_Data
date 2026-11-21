"""
Shared code for the four method scripts (1_rule_based.py ... 4_code_mcp.py).

  * text-only Hugging Face model loading (no vision / audio encoders) + batched generation
  * data loading from data/real/ (same schema as data/sample/), entity scoping, token budgets
  * scoring: correct / peek / trap / false_abstain / wrong (identical to the notebooks)
  * per-question audit files: responses/<model>/<method>/<variant>/<qid>.json
  * per-question tables:      results/<model>/<method>/<variant>.csv (+ _meta.json with cost and wall time)
  * summaries:                results/summary_<method>.csv (accuracy, cost, time)
  * fact store shared by methods 3 and 4 (LLM extraction or oracle), cached per model

Run a method script, not this file.
"""
import argparse, collections, datetime as dt, gc, glob, json, os, random, re, statistics, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))     # .../pit_pipeline/methods
ROOT = os.path.dirname(HERE)                           # .../pit_pipeline
SOURCES_ALL = ["broker", "alfred", "edgar"]

# ====================================================================================== models (TEXT-ONLY by default)
# Qwen3.5 and Gemma 4 checkpoints are multimodal. TEXT_ONLY = True loads just the language model
# (AutoModelForCausalLM on the checkpoint's text_config, weights remapped from model.language_model.*),
# so the vision / audio encoders are never instantiated. Phi-4-mini is text-only already.
TEXT_ONLY = True

MODELS = {
    "qwen3.5-4b": dict(hf_id="Qwen/Qwen3.5-4B", thinking=True,
                       sample_think=dict(temperature=1.0, top_p=0.95, top_k=20),
                       sample_nothink=dict(temperature=0.7, top_p=0.8, top_k=20)),
    "gemma-4-e4b-it": dict(hf_id="google/gemma-4-E4B-it", thinking=True,
                           sample_think=dict(temperature=1.0, top_p=0.95, top_k=64),
                           sample_nothink=dict(temperature=1.0, top_p=0.95, top_k=64)),
    "phi-4-mini": dict(hf_id="microsoft/Phi-4-mini-instruct", thinking=False,      # no thinking mode
                       sample_think=None,
                       sample_nothink=dict(temperature=0.7, top_p=0.95)),
}

# checkpoint key prefixes of the language model inside multimodal checkpoints -> text-only class names
TEXT_KEY_MAP = {r"^model\.language_model\.": "model.",
                r"^language_model\.model\.": "model.",
                r"^language_model\.lm_head\.": "lm_head."}
ENCODER_NAMES = re.compile(r"^(visual|vision_tower|vision_model|vision_encoder|audio_tower|audio_model|audio_encoder|"
                           r"embed_vision|embed_audio|multi_modal_projector|mm_projector)$")


class S:
    """Run state shared by the helpers."""
    args = None
    llm = None          # TextLM, or None in --dry-run
    model = "dry_run"   # model key
    seed = 0            # rerun index (--runs)
    backend = "dry"     # vllm | hf | dry


# ====================================================================================== CLI
def common_args(description):
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data and output")
    g.add_argument("--data-dir", default=os.path.join(ROOT, "data", "real"), help="folder with <source>_evidence.csv, _questions.csv, _rules.md")
    g.add_argument("--sources", nargs="+", default=SOURCES_ALL, choices=SOURCES_ALL)
    g.add_argument("--entities", nargs="+", default=None, help="only these entities (e.g. AAPL MSFT US_GDP)")
    g.add_argument("--limit", type=int, default=0, help="random sample of N questions per source (0 = all)")
    g.add_argument("--limit-seed", type=int, default=0)
    g.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    g.add_argument("--responses-dir", default=os.path.join(ROOT, "responses"), help="one JSON per question (audit trail)")
    g.add_argument("--overwrite", action="store_true", help="re-run questions that already have a response file (default: resume)")
    g = p.add_argument_group("models")
    g.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    g.add_argument("--dry-run", action="store_true", help="no model; dummy answers (checks the plumbing)")
    g.add_argument("--with-encoders", action="store_true", help="load the full multimodal checkpoint (default: text-only)")
    g.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    g.add_argument("--load-4bit", action="store_true", help="bitsandbytes NF4 for small GPUs")
    g.add_argument("--attn", default=None, help="attn_implementation, e.g. sdpa or flash_attention_2")
    g = p.add_argument_group("decoding")
    g.add_argument("--thinking", choices=["off", "on", "both"], default="off", help="Qwen3.5 / Gemma 4 only")
    g.add_argument("--think-budget", type=int, default=8000, help="extra new tokens when thinking")
    g.add_argument("--sample", action="store_true", help="model-card sampling instead of greedy")
    g.add_argument("--runs", type=int, default=1, help="reruns (seeds 0..N-1, use with --sample) for stability")
    g = p.add_argument_group("inference engine (the same engine and settings must be used for every method)")
    g.add_argument("--backend", choices=["auto", "vllm", "hf"], default="auto",
                   help="auto = vllm if installed (much faster: continuous batching, paged attention), else hf")
    g.add_argument("--gpu-mem", type=float, default=0.88, help="vllm gpu_memory_utilization")
    g.add_argument("--prefix-cache", action="store_true",
                   help="vllm prefix caching (off by default so no method gets cache hits from repeated prompts)")
    g.add_argument("--tp", type=int, default=1, help="vllm tensor-parallel GPUs")
    g.add_argument("--latency-sample", type=int, default=30,
                   help="after each variant, time this many fixed questions one at a time (same qids for every method); 0 = off")
    g = p.add_argument_group("throughput and context")
    g.add_argument("--batch-size", type=int, default=64, help="hf: max sequences per generate() call")
    g.add_argument("--batch-tokens", type=int, default=0,
                   help="hf: max (prompt + new tokens) x batch size; 0 = auto from free GPU memory and KV size")
    g.add_argument("--max-input-tokens", type=int, default=32768,
                   help="prompt budget; oldest evidence/fact lines are dropped beyond it (logged as n_dropped)")
    g.add_argument("--chunk", type=int, default=256, help="questions per checkpoint chunk")
    return p


# ====================================================================================== text-only LM
THINK_BLOCKS = [(r"<think>", r"</think>"), (r"<\|channel>", r"<channel\|>")]   # Qwen / Gemma 4
EXTRA_SPECIAL = ["<turn|>", "<|turn>", "<|im_end|>", "<|im_start|>", "<|end|>", "<|endoftext|>",
                 "<eos>", "<bos>", "<end_of_turn>", "<|assistant|>"]


class _ChatBase:
    """Chat template rendering, token counting and output cleaning shared by both engines."""
    backend = "?"

    def render(self, msgs, thinking):
        kw = dict(tokenize=False, add_generation_prompt=True)
        if self.cfg["thinking"]:
            kw["enable_thinking"] = bool(thinking)
        return self.tok.apply_chat_template(msgs, **kw)

    def count_many(self, texts):
        return [len(x) for x in self.tok(list(texts), add_special_tokens=False).input_ids] if texts else []

    def clean(self, raw):
        txt = raw
        for a, b in THINK_BLOCKS:
            txt = re.sub(a + r".*?" + b, "", txt, flags=re.S)
            txt = re.split(b, txt)[-1]
        for a, _ in THINK_BLOCKS:            # unclosed thinking (ran out of tokens): nothing usable
            txt = re.split(a, txt)[0]
        for t in self.special:
            txt = txt.replace(t, "")
        return txt.strip()

    def _set_special(self):
        self.special = sorted({t for t in (self.tok.all_special_tokens or []) if t} | set(EXTRA_SPECIAL), key=len, reverse=True)

    def reset_cache(self):
        pass


class TextLM(_ChatBase):
    backend = "hf"
    def __init__(self, key, a):
        import torch, transformers as T
        self.torch, self.T, self.cfg, self.key, self.a = torch, T, MODELS[key], key, a
        hf_id = self.cfg["hf_id"]
        print(f"[load] {hf_id}  (transformers {T.__version__})", flush=True)
        self.tok = T.AutoTokenizer.from_pretrained(hf_id)
        if not getattr(self.tok, "chat_template", None):            # template shipped with the processor only
            proc = T.AutoProcessor.from_pretrained(hf_id)
            self.tok.chat_template = proc.chat_template
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model, self.load_note = self._load(hf_id)
        self.model.eval()
        self.device = self.model.device
        gc_eos = self.model.generation_config.eos_token_id
        self.eos = set(gc_eos if isinstance(gc_eos, list) else [gc_eos]) | {self.tok.eos_token_id}
        self.eos.discard(None)
        n = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(f"[load] {self.load_note}; {n:.2f}B parameters on {self.device}", flush=True)
        self._set_special()
        self.batch_tokens = a.batch_tokens or self._auto_batch_tokens()
        print(f"[load] hf generate: attention={self.attn}, batch_size<={a.batch_size}, batch_tokens<={self.batch_tokens}", flush=True)

    def _auto_batch_tokens(self):
        """Tokens of KV cache that fit in ~60% of free GPU memory (OOM batches are split anyway)."""
        torch = self.torch
        if not torch.cuda.is_available():
            return 65536
        c = getattr(self.model.config, "text_config", None) or self.model.config
        heads = getattr(c, "num_attention_heads", 32)
        kv = getattr(c, "num_key_value_heads", None) or heads
        hd = getattr(c, "head_dim", None) or c.hidden_size // heads
        per_tok = 2 * c.num_hidden_layers * kv * hd * 2          # K and V, bf16
        free = torch.cuda.mem_get_info()[0]
        return int(max(16384, min(2_000_000, 0.6 * free / per_tok)))

    # ---------- loading
    def _kw(self):
        torch, T, a = self.torch, self.T, self.a
        kw = dict(device_map="auto", dtype=getattr(torch, a.dtype))
        if a.load_4bit:
            kw["quantization_config"] = T.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                             bnb_4bit_compute_dtype=getattr(torch, a.dtype))
        if a.attn:
            self.attn = a.attn
        else:   # flash-attention handles left padding without materialising masks; much faster on long prompts
            import importlib.util
            self.attn = "flash_attention_2" if importlib.util.find_spec("flash_attn") and torch.cuda.is_available() else "sdpa"
        kw["attn_implementation"] = self.attn
        return kw

    def _check(self, model, info):
        missing = list(info.get("missing_keys", []))
        other = [k for k in missing if "lm_head" not in k]
        if other:
            raise RuntimeError(f"{len(other)} weights not found in the checkpoint, e.g. {other[:3]}")
        if missing:   # lm_head missing is fine only if it is tied to the input embeddings
            o, i = model.get_output_embeddings(), model.get_input_embeddings()
            if o is None or i is None or o.weight.data_ptr() != i.weight.data_ptr():
                raise RuntimeError("lm_head not in checkpoint and not tied to the embeddings")

    def _free(self):
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def _load(self, hf_id):
        T, kw, errors = self.T, self._kw(), []
        cfg = T.AutoConfig.from_pretrained(hf_id)
        tcfg = getattr(cfg, "text_config", None)
        if tcfg is None:                                   # Phi-4-mini: plain causal LM
            model, info = T.AutoModelForCausalLM.from_pretrained(hf_id, output_loading_info=True, **kw)
            self._check(model, info)
            return model, f"text-only {type(model).__name__}"
        if TEXT_ONLY and not self.a.with_encoders:
            # 1) text-only class built from text_config; only the language-model weights are read
            if getattr(cfg, "tie_word_embeddings", None) is not None:
                tcfg.tie_word_embeddings = cfg.tie_word_embeddings
            try:
                model, info = T.AutoModelForCausalLM.from_pretrained(
                    hf_id, config=tcfg, key_mapping=TEXT_KEY_MAP, output_loading_info=True, **kw)
                self._check(model, info)
                return model, f"text-only {type(model).__name__} (vision/audio weights not loaded)"
            except Exception as e:
                errors.append(f"text_config route: {e}")
                model = None; self._free()
        # 2) full checkpoint; with text-only on, the encoders are deleted right after loading
        for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForMultimodalLM"):
            cls = getattr(T, name, None)
            if cls is None:
                continue
            try:
                model, info = cls.from_pretrained(hf_id, output_loading_info=True, **kw)
                self._check(model, info)
            except Exception as e:
                errors.append(f"{name}: {e}"); self._free(); continue
            if self.a.with_encoders:
                return model, f"full multimodal {type(model).__name__} (--with-encoders)"
            dropped = []
            for parent in (model, getattr(model, "model", None)):
                for child, _ in list(parent.named_children()) if parent is not None else []:
                    if ENCODER_NAMES.match(child):
                        setattr(parent, child, None); dropped.append(child)
            self._free()
            return model, f"{type(model).__name__} with encoders removed {dropped} (text-only fallback; {errors[:1]})"
        raise RuntimeError(f"could not load {hf_id}; upgrade transformers (pip install -U transformers). " + " | ".join(errors))

    # ---------- generation
    def _gen_kwargs(self, max_new, thinking):
        g = dict(max_new_tokens=max_new, pad_token_id=self.tok.pad_token_id)
        if self.a.sample:
            samp = self.cfg["sample_think" if thinking else "sample_nothink"] or self.cfg["sample_nothink"]
            g.update(do_sample=True, **samp)
            self.torch.manual_seed(S.seed)
        else:
            g.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        return g

    def generate(self, convs, max_new, thinking):
        prompts = [self.render(m, thinking) for m in convs]
        ids = self.tok(prompts, add_special_tokens=False).input_ids
        order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))      # longest first, little padding
        out, cur = [None] * len(ids), []
        for i in order:
            longest = len(ids[cur[0]]) if cur else len(ids[i])
            if cur and (len(cur) >= self.a.batch_size or (len(cur) + 1) * (longest + max_new) > self.batch_tokens):
                self._batch(cur, ids, max_new, thinking, out); cur = []
            cur.append(i)
        if cur:
            self._batch(cur, ids, max_new, thinking, out)
        return out

    def _batch(self, idx, ids, max_new, thinking, out):
        torch = self.torch
        try:
            enc = self.tok.pad({"input_ids": [ids[i] for i in idx]}, padding=True, return_tensors="pt").to(self.device)
            t = time.time()
            with torch.inference_mode():
                gen = self.model.generate(**enc, **self._gen_kwargs(max_new, thinking))
            sec = (time.time() - t) / len(idx)        # wall time of the batch, shared equally
        except torch.cuda.OutOfMemoryError:
            if len(idx) == 1:
                raise
            self._free(); h = len(idx) // 2
            print(f"    [oom] splitting batch of {len(idx)}", flush=True)
            self._batch(idx[:h], ids, max_new, thinking, out); self._batch(idx[h:], ids, max_new, thinking, out)
            return
        width = enc["input_ids"].shape[1]
        for j, i in enumerate(idx):
            new = gen[j, width:].tolist()
            n = next((k + 1 for k, t in enumerate(new) if t in self.eos), len(new))
            raw = self.tok.decode(new[:n], skip_special_tokens=False)
            out[i] = dict(text=self.clean(raw), raw=raw, in_tokens=len(ids[i]), out_tokens=n, seconds=sec)

    def close(self):
        del self.model
        self._free()


# ====================================================================================== vLLM engine (offline, in-process)
class VLLMLM(_ChatBase):
    """Same interface as TextLM. vLLM schedules all requests of a call itself (continuous batching, paged
    attention), so long prompts no longer collapse the batch. Text-only: language_model_only (skips the
    encoders) or, on older vLLM, zero multimodal inputs per prompt."""
    backend = "vllm"

    def __init__(self, key, a):
        short_tmpdir()
        import torch, transformers as T, vllm
        from vllm import LLM
        self.torch, self.cfg, self.key, self.a = torch, MODELS[key], key, a
        hf_id = self.cfg["hf_id"]
        print(f"[load] {hf_id}  (vllm {vllm.__version__})", flush=True)
        hcfg = T.AutoConfig.from_pretrained(hf_id)
        tcfg = getattr(hcfg, "text_config", None) or hcfg
        ctx = max_context_needed(a)
        model_max = getattr(tcfg, "max_position_embeddings", None) or ctx
        self.max_len = min(ctx, model_max)
        kw = dict(model=hf_id, dtype=a.dtype, max_model_len=self.max_len, gpu_memory_utilization=a.gpu_mem,
                  enable_prefix_caching=a.prefix_cache, tensor_parallel_size=a.tp, seed=0)
        if a.load_4bit:
            kw["quantization"] = "bitsandbytes"
        attempts = [({}, "text-only model")]
        if hasattr(hcfg, "text_config") and TEXT_ONLY and not a.with_encoders:
            attempts = [(dict(language_model_only=True), "text-only (language_model_only: encoders not loaded)"),
                        (dict(limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0}), "text-only (0 image/video/audio per prompt)"),
                        (dict(limit_mm_per_prompt={"image": 0, "video": 0}), "text-only (0 image/video per prompt)")]
        errors = []
        for extra, note in attempts:
            try:
                self.llm = LLM(**kw, **extra)
                self.load_note = f"vllm {note}, max_model_len={self.max_len}, prefix_cache={a.prefix_cache}"
                break
            except Exception as e:      # unknown engine arg / unsupported modality: try the next text-only option
                errors.append(f"{list(extra)}: {str(e)[:200]}")
                gc.collect(); torch.cuda.empty_cache()
        else:
            raise RuntimeError("vllm could not load the model: " + " | ".join(errors))
        self.tok = self.llm.get_tokenizer()
        self._set_special()
        print(f"[load] {self.load_note}", flush=True)

    def _params(self, max_new, thinking):
        from vllm import SamplingParams
        if self.a.sample:
            samp = self.cfg["sample_think" if thinking else "sample_nothink"] or self.cfg["sample_nothink"]
            return dict(max_tokens=max_new, seed=S.seed, skip_special_tokens=False, **samp)
        return dict(max_tokens=max_new, temperature=0.0, skip_special_tokens=False)

    def generate(self, convs, max_new, thinking):
        from vllm import SamplingParams
        ids = self.tok([self.render(m, thinking) for m in convs], add_special_tokens=False).input_ids
        base = self._params(max_new, thinking)
        ok = [i for i, x in enumerate(ids) if len(x) < self.max_len - 16]
        prompts = [{"prompt_token_ids": ids[i]} for i in ok]
        params = [SamplingParams(**dict(base, max_tokens=min(max_new, self.max_len - len(ids[i])))) for i in ok]
        t = time.time()
        outs = self.llm.generate(prompts, params, use_tqdm=False) if prompts else []
        sec = (time.time() - t) / max(1, len(ok))          # wall time of the call, shared equally (as hf)
        res = [dict(text="", raw=f"[prompt of {len(x)} tokens exceeds max_model_len {self.max_len}]",
                    in_tokens=len(x), out_tokens=0, seconds=0.0) for x in ids]
        for i, o in zip(ok, outs):
            c = o.outputs[0]
            res[i] = dict(text=self.clean(c.text), raw=c.text, in_tokens=len(ids[i]), out_tokens=len(c.token_ids), seconds=sec)
        return res

    def reset_cache(self):
        if self.a.prefix_cache:
            try: self.llm.reset_prefix_cache()
            except Exception: pass

    def close(self):
        del self.llm
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
            destroy_model_parallel(); destroy_distributed_environment()
        except Exception:
            pass
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def short_tmpdir():
    """vLLM creates zmq IPC sockets in the temp dir; Unix socket paths must be < 108 chars (sockaddr_un)."""
    import tempfile
    if len(tempfile.gettempdir()) > 60:
        print(f"[engine] TMPDIR {tempfile.gettempdir()} is too long for IPC sockets; using /tmp", flush=True)
        os.environ["TMPDIR"] = "/tmp"
        tempfile.tempdir = None
    os.environ.setdefault("VLLM_RPC_BASE_PATH", tempfile.gettempdir())


def max_context_needed(a):
    """Prompt budget + answer + thinking + slack (method 4 conversations stay well inside)."""
    think = a.think_budget if a.thinking != "off" else 0
    return a.max_input_tokens + 1500 + think + 2048


def resolve_backend(a):
    if a.dry_run:
        return "dry"
    import importlib.util, importlib.metadata
    want = a.backend
    if want == "auto":
        want = "vllm" if importlib.util.find_spec("vllm") else "hf"
    if want == "vllm":   # Qwen3.5 needs vLLM >= 0.17; Gemma 4 a recent release: fail early with a clear message
        try:
            v = importlib.metadata.version("vllm")
        except importlib.metadata.PackageNotFoundError:
            sys.exit("vllm is not installed in this Python. Run: bash setup_env.sh && source env.sh")
        major_minor = tuple(int(x) for x in re.findall(r"\d+", v)[:2])
        if major_minor < (0, 17):
            sys.exit(f"vllm {v} is too old for Qwen3.5 / Gemma 4 (need >= 0.17; latest recommended). "
                     f"Run: bash setup_env.sh && source env.sh   (or pass --backend hf)")
    return want


def make_engine(key, a):
    return VLLMLM(key, a) if S.backend == "vllm" else TextLM(key, a)


# ====================================================================================== LLM entry points
DUMMY = '{"answer": "CANNOT_ANSWER", "evidence_ids": []}'


def gen(convs, max_tokens=1000, thinking=False):
    """Batched chat completion. convs: list of message lists (system first). Returns list of dicts."""
    if not convs:
        return []
    if S.llm is None:
        return [dict(text=DUMMY, raw="", in_tokens=sum(ntok([m["content"] for m in c])), out_tokens=0, seconds=0.0)
                for c in convs]
    return S.llm.generate(convs, max_tokens + (S.args.think_budget if thinking else 0), thinking)


def ntok(texts):
    texts = list(texts)
    return S.llm.count_many(texts) if S.llm is not None else [len(t) // 4 + 1 for t in texts]


def fit_lines(lines, budget, counts=None):
    """Keep the most recent lines that fit in `budget` tokens. Returns (kept_lines, n_dropped)."""
    if not lines:
        return lines, 0
    counts = counts if counts is not None else ntok(lines)
    total, k = 0, len(lines)
    for c in reversed(counts):
        if total + c + 1 > budget:
            break
        total += c + 1; k -= 1
    return lines[k:], k


def think_values():
    vals = {"off": [False], "on": [True], "both": [False, True]}[S.args.thinking]
    if S.model in MODELS and not MODELS[S.model]["thinking"]:
        vals = [v for v in vals if not v]
    return vals or [False]


def ttag(thinking):
    return "think" if thinking else "nothink"


# ====================================================================================== data
def line(r):
    return f"[{r.evidence_id}] {r.known_at:%Y-%m-%d %H:%M} | {r.text}"


class Source:
    """Evidence, questions and rules of one benchmark, scoped to the entities being asked about."""

    def __init__(self, name):
        a, d = S.args, S.args.data_dir
        self.name = name
        ev = pd.read_csv(f"{d}/{name}_evidence.csv")
        ev["known_at"] = pd.to_datetime(ev["known_at"])
        qs = pd.read_csv(f"{d}/{name}_questions.csv", keep_default_na=False, dtype=str)
        if a.entities:
            qs = qs[qs.entity.isin(a.entities)]
        if a.limit and len(qs) > a.limit:
            qs = qs.sample(a.limit, random_state=a.limit_seed).sort_index()
        self.qs = qs.reset_index(drop=True)
        self.ev = ev[ev.entity.isin(set(self.qs.entity))].sort_values(["entity", "known_at", "evidence_id"]).reset_index(drop=True)
        self.rules = open(f"{d}/{name}_rules.md").read()
        self.known = dict(zip(self.ev.evidence_id, self.ev.known_at))
        self.by_entity = {e: g for e, g in self.ev.groupby("entity", sort=False)}
        self._lines, self._counts = {}, {}

    def entity_lines(self, entity):
        """Evidence lines of one entity (chronological) and their token counts, cached."""
        if entity not in self._lines:
            g = self.by_entity.get(entity, self.ev.iloc[:0])
            self._lines[entity] = (g, [line(r) for r in g.itertuples()])
            self._counts[entity] = ntok(self._lines[entity][1])
        g, lines = self._lines[entity]
        return g, lines, self._counts[entity]


def load_sources():
    out = {}
    for s in S.args.sources:
        for suffix in ("_evidence.csv", "_questions.csv", "_rules.md"):
            if not os.path.exists(f"{S.args.data_dir}/{s}{suffix}"):
                sys.exit(f"missing {S.args.data_dir}/{s}{suffix}  (use --data-dir)")
        src = Source(s)
        if len(src.qs):
            out[s] = src
            print(f"[data] {s}: {len(src.qs)} questions, {src.ev.entity.nunique()} entities, {len(src.ev)} evidence lines", flush=True)
    return out


# ====================================================================================== prompts and parsing
ANSWER_SYSTEM = """You answer questions about financial data that keeps being revised and corrected.
Answer as of the given as-of time: anything published after it does not exist.
Rules for this dataset:
<<RULES>>
Reply with JSON only: {"answer": <number, short text, or "CANNOT_ANSWER">, "evidence_ids": [<ids you relied on>]}
Numbers: a plain number in the unit asked (no $, %, or commas)."""


def answer_system(rules):
    return ANSWER_SYSTEM.replace("<<RULES>>", rules)


ANSWER_RE = re.compile(r'"answer"\s*:\s*("(?:[^"\\]|\\.)*"|-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE]-?\d+)?|true|false|null)', re.S)
IDS_RE = re.compile(r'"evidence_ids"\s*:\s*\[(.*)', re.S)


def parse(txt, with_flag=False):
    """{"answer": ..., "evidence_ids": [...]} from a reply. If the JSON is cut off (smoke test: Phi-4-mini listed
    hundreds of evidence ids and hit the output limit after a complete "answer"), the answer and the ids
    written so far are still recovered, so a truncated citation list is not scored as no answer."""
    txt = re.sub(r"```(?:json)?", "", txt)
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        d = json.loads(m.group()) if m else {}
    except Exception:
        d = {}
    if not isinstance(d, dict):
        d = {}
    if "answer" not in d:                       # salvage a truncated / malformed object
        a = ANSWER_RE.search(txt)
        if a:
            try:
                d["answer"] = json.loads(a.group(1))
            except Exception:
                d["answer"] = a.group(1).strip('"')
            i = IDS_RE.search(txt[a.end():] if IDS_RE.search(txt[a.end():]) else txt)
            if i:
                d["evidence_ids"] = re.findall(r'"([^"\s]+)"', i.group(1).split("]")[0])
    ids = d.get("evidence_ids", []) or []
    ids = ids if isinstance(ids, list) else [ids]
    ans = d.get("answer", txt.strip()[:200])
    ans = "" if ans is None else ans
    out = str(ans), [str(x) for x in ids]
    return (*out, "answer" in d) if with_flag else out


# ====================================================================================== scoring (as notebooks)
def norm(s):
    return " ".join(re.sub(r"[^a-z0-9; ]", " ", str(s).lower()).split())


def to_num(s):
    m = re.search(r"-?\d[\d,]*\.?\d*", str(s).replace("+", ""))
    return float(m.group().replace(",", "")) if m else None


def match(ans, target, atype, tol):
    if target == "":
        return False
    if target == "CANNOT_ANSWER":
        return "cannot" in norm(ans)
    if "cannot" in norm(ans):
        return False
    if atype == "number":
        a = to_num(ans)
        return a is not None and abs(a - float(target)) <= float(tol) + 1e-9
    return all(norm(t) in norm(ans) for t in target.split(";"))


def score(q, ans, cited, known):
    if match(ans, q.gold, q.answer_type, q.tol):                                     cat = "correct"
    elif any(match(ans, p, q.answer_type, q.tol) for p in q.peek_answers.split("|")): cat = "peek"
    elif any(match(ans, p, q.answer_type, q.tol) for p in q.trap_answers.split("|")): cat = "trap"
    elif "cannot" in norm(ans):                                                       cat = "false_abstain"
    else:                                                                             cat = "wrong"
    cut = pd.Timestamp(q.as_of)
    future = [c for c in cited if c in known and known[c] > cut]
    gold_ev = [g for g in q.gold_evidence.split("|") if g]
    recall = len(set(gold_ev) & set(cited)) / len(gold_ev) if gold_ev else None
    return dict(category=cat, correct=cat == "correct", cited_future=bool(future), evidence_recall=recall)


# ====================================================================================== per-question audit + run loop
BIG_KEYS = ("prompt", "raw_output", "final_text", "transcript")


def _safe(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def variant_name(v):
    return f"{v}_r{S.seed}" if S.args.runs > 1 else v


def qpath(method, variant, qid):
    return os.path.join(S.args.responses_dir, S.model, method, variant, _safe(qid) + ".json")


def finalize(method, variant, src, q, res, messages, extra=None):
    """Score one answer, write its audit JSON, return its table row."""
    ans, cited, json_ok = parse(res["text"], with_flag=True)
    sc = score(q, ans, cited, src.known)
    rec = dict(qid=q.qid, backend=S.backend, model=S.model, hf_id=MODELS.get(S.model, {}).get("hf_id", ""), method=method, variant=variant,
               source=src.name, entity=q.entity, qtype=q.qtype, mode=q.mode, as_of=q.as_of, question=q.question,
               gold=q.gold, peek_answers=q.peek_answers, trap_answers=q.trap_answers, gold_evidence=q.gold_evidence,
               answer=ans, cited="|".join(cited), json_ok=json_ok, **sc,
               in_tokens=res["in_tokens"], out_tokens=res["out_tokens"], seconds=round(res["seconds"], 4),
               **(extra or {}), created=dt.datetime.now().isoformat(timespec="seconds"),
               prompt=messages, final_text=res["text"], raw_output=res["raw"])
    p = qpath(method, variant, q.qid)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    return {k: v for k, v in rec.items() if k not in BIG_KEYS}


def load_row(p):
    with open(p) as f:
        rec = json.load(f)
    return {k: v for k, v in rec.items() if k not in BIG_KEYS}


def single_turn(items, thinking, max_tokens=1000):
    """Default solver: one batched call per question. items: (src, q, messages, extra)."""
    outs = gen([m for _, _, m, _ in items], max_tokens, thinking)
    return [(o, m, e) for (_, _, m, e), o in zip(items, outs)]


def run_variant(method, variant, sources, build, thinking, solve=None, ingestion=None, settings=None):
    """build(src, q) -> (messages, extra). solve(items, thinking) -> [(result, messages_or_transcript, extra)]."""
    variant = variant_name(variant)
    solve = solve or single_turn
    print(f"\n[{S.model}] {method} / {variant}", flush=True)
    rows, t_start = [], time.time()
    for src in sources.values():
        todo = []
        for q in src.qs.itertuples():
            p = qpath(method, variant, q.qid)
            if os.path.exists(p) and not S.args.overwrite:
                rows.append(load_row(p))
            else:
                todo.append(q)
        if len(todo) < len(src.qs):
            print(f"  {src.name}: {len(src.qs) - len(todo)} answered before (resume), {len(todo)} to go", flush=True)
        for i in range(0, len(todo), S.args.chunk):
            chunk = todo[i:i + S.args.chunk]
            items = [(src, q, *build(src, q)) for q in chunk]
            for (s_, q, _, _), (res, msgs, extra) in zip(items, solve(items, thinking)):
                rows.append(finalize(method, variant, s_, q, res, msgs, extra))
            done = rows[-len(chunk):]
            acc = np.mean([r["correct"] for r in done])
            print(f"  {src.name}: {min(i + len(chunk), len(todo))}/{len(todo)}  chunk accuracy {acc:.2f}", flush=True)
    wall = time.time() - t_start
    lat = latency_probe(sources, build, solve, thinking)
    df = pd.DataFrame(rows)
    d = os.path.join(S.args.out_dir, S.model, method)
    os.makedirs(d, exist_ok=True)
    df.to_csv(f"{d}/{variant}.csv", index=False)
    meta = dict(backend=S.backend, gpu=gpu_name(), model=S.model, hf_id=MODELS.get(S.model, {}).get("hf_id", ""),
                method=method, variant=variant,
                loader=getattr(S.llm, "load_note", "dry run"), thinking=thinking, wall_seconds_answering=round(wall, 2),
                latency_qids=[x[0] for x in lat], latency_seconds=[x[1] for x in lat],
                ingestion=ingestion or {}, settings=settings or {}, data_dir=S.args.data_dir,
                args={k: v for k, v in vars(S.args).items() if k not in ("models",)},
                finished=dt.datetime.now().isoformat(timespec="seconds"))
    json.dump(meta, open(f"{d}/{variant}_meta.json", "w"), indent=1, default=str)
    if len(df):
        print(f"  -> accuracy {df.correct.mean():.3f} over {len(df)} questions; {d}/{variant}.csv", flush=True)
        nd = pd.to_numeric(df.get("n_dropped", pd.Series(dtype=float)), errors="coerce").fillna(0)
        if (nd > 0).any():
            print(f"  !! {int((nd > 0).sum())} prompts were truncated to --max-input-tokens {S.args.max_input_tokens}; "
                  f"cap the history further (run_pipeline.sh HISTORY_TOKENS) so every method sees all evidence", flush=True)
    return df


def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "n/a"


def latency_qids(sources):
    """The same questions for every method, model and variant: seeded sample of the selected qids per source."""
    n = S.args.latency_sample
    if n <= 0:
        return []
    per = -(-n // len(sources))
    out = []
    for src in sources.values():
        qids = sorted(src.qs.qid)
        out += [(src, q) for q in src.qs.itertuples()
                if q.qid in set(random.Random(f"latency-{src.name}").sample(qids, min(per, len(qids))))]
    return out


def latency_probe(sources, build, solve, thinking):
    """Batch-independent timing: each sampled question alone (batch of 1), end to end incl. tool turns.
    Answers are not re-scored; the throughput run above is the scored one."""
    if S.llm is None:
        return []
    picks, out = latency_qids(sources), []
    if picks:
        print(f"  latency probe: {len(picks)} questions, one at a time", flush=True)
    for src, q in picks:
        msgs, extra = build(src, q)
        S.llm.reset_cache()
        t = time.time()
        solve([(src, q, msgs, extra)], thinking)
        out.append((q.qid, round(time.time() - t, 4)))
    return out


# ====================================================================================== per-entity sequential LLM work, batched across entities
def lockstep(streams, make_conv, update, max_tokens, thinking=False, label=""):
    """streams: {entity: [rows]} processed in order within each entity; step k of all entities is one batch.
    make_conv(entity, row) -> messages; update(entity, row, result). Returns cost dict."""
    cost = dict(in_tokens=0, out_tokens=0, seconds=0.0, llm_calls=0)
    pos = {k: 0 for k in streams}
    total, done, t0, last = sum(len(v) for v in streams.values()), 0, time.time(), 0
    while True:
        keys = [k for k in streams if pos[k] < len(streams[k])]
        if not keys:
            break
        convs = [make_conv(k, streams[k][pos[k]]) for k in keys]
        outs = gen(convs, max_tokens, thinking)
        for k, o in zip(keys, outs):
            update(k, streams[k][pos[k]], o)
            pos[k] += 1
            cost["in_tokens"] += o["in_tokens"]; cost["out_tokens"] += o["out_tokens"]
            cost["seconds"] += o["seconds"]; cost["llm_calls"] += 1
        done += len(keys)
        if done - last >= 500 or done == total:
            last = done
            print(f"    {label} {done}/{total} evidence lines ({time.time() - t0:.0f}s)", flush=True)
    return cost


# ====================================================================================== fact store (methods 3 and 4)
EXTRACT_SYSTEM = """You turn one piece of evidence into facts for a bitemporal knowledge graph.
Use exactly the fact schema in the rules (subject, attribute, period, value, effective_at, kind).
Use the facts already in the graph to resolve names (renames) and to detect revisions, corrections and missed revisions.
Rules for this dataset:
<<RULES>>
Return JSON only: a list of objects with keys subject, attribute, period, value, effective_at ("YYYY-MM-DD HH:MM:SS" or ""), kind."""
FACT_COLS = ["evidence_id", "learned_at", "subject", "attribute", "period", "value", "effective_at", "kind"]


def state_lines(facts, limit=40):
    """Latest fact per (subject, attribute, period) of one entity, shown to the extractor."""
    if not facts:
        return "(none yet)"
    latest = {}
    for x in facts:
        latest[(x["subject"], x["attribute"], x["period"])] = x
    rows = list(latest.values())[-limit:]
    return "\n".join(f"{x['subject']} | {x['attribute']} | {x['period'] or '-'} = {x['value']} "
                     f"(effective {x['effective_at'] or '-'}, {x['evidence_id']})" for x in rows)


def canon_subject(subject, entity, attribute=""):
    """Attach an LLM-extracted subject to the evidence line's entity.
    Run 1 lost facts here: models copied the rules card's sample entity (ZETA_GDP) or wrote 'AAPL \\| Barclays',
    and entity_facts() (subject == entity or startswith 'entity |') then dropped them. The evidence line tells
    us the entity, so the first subject part is replaced by it; the rest (broker / series / concept) is kept."""
    s = re.sub(r"\s*\\?\|\s*", " | ", str(subject).replace("\\|", "|")).strip(" |")
    parts = [x for x in s.split(" | ") if x]
    if not parts:
        return entity
    if parts[0].upper() == entity.upper() or len(parts) > 1:
        parts = parts[1:]
    elif str(attribute).strip() == "split_ratio":   # a split belongs to the ticker itself
        parts = []
    # else: a single part that is not the entity is a name without the ticker (e.g. 'Barclays') -> keep it
    # The schema has at most one part after the entity (broker / series name / concept). Weak models (Phi-4-mini
    # in the smoke test) wrote the whole fact into the subject ('US_GDP | US real GDP growth | estimate | 2018Q2
    # = 4.2'), giving every fact its own subject; keep the first part only, without any '= value' / '(...)' tail.
    parts = [re.split(r"\s*(?:=|\(|:)\s*", x)[0].strip() for x in parts]
    parts = [x for x in parts if x and not _junk_part(x, attribute)][:1]
    return " | ".join([entity] + parts)


ATTR_NAMES = {"price_target_usd", "rating", "split_ratio", "alias", "estimate", "value_usd_m"}


def _junk_part(x, attribute=""):
    """A subject part that is not a name: an evidence id (US_GDP-00284), an attribute, a period, a number."""
    return bool(re.fullmatch(r"[A-Za-z0-9_.]+-\d{3,}", x) or x in ATTR_NAMES or x == str(attribute).strip()
                or re.fullmatch(r"[\d\s.,%$+\-]+|\d{4}(?:Q\d|-\d\d.*|\.\..*)?", x))


SINGLE_SERIES = {"alfred": "estimate", "edgar": "value_usd_m"}   # one series (one attribute) per entity


def normalize_facts(rows, source, entity):
    """Code-side entity resolution of LLM-extracted facts (same for every model; oracle facts already conform).
    alfred / edgar have ONE series per entity, so all of an entity's facts get its most common subject and the
    single attribute (Phi-4-mini wrote evidence ids into subjects and left attributes empty, splitting one
    series into hundreds of groups). Reiteration wordings become kind 'corroborate'."""
    if not rows:
        return rows
    if source in SINGLE_SERIES:
        named = [x["subject"] for x in rows if x["subject"] != entity]
        main = collections.Counter(named).most_common(1)[0][0] if named else entity
        for x in rows:
            x["subject"], x["attribute"] = main, SINGLE_SERIES[source]
    for x in rows:
        k = str(x.get("kind", "")).strip().lower()
        if "reiter" in k or "unchanged" in k or k in ("maintain", "maintained", "same", "repeat"):
            x["kind"] = "corroborate"
        elif "missed" in k:
            x["kind"] = "inferred"
    return rows


def _parse_facts(txt):
    m = re.search(r"\[.*\]", txt or "", re.S)
    try:
        items = json.loads(m.group()) if m else []
    except Exception:
        items = []
    items = [items] if isinstance(items, dict) else items
    return [it for it in items if isinstance(it, dict)]


def extract_llm(src, cache_dir):
    """One LLM call per evidence line, in time order within each entity; entities run in parallel batches.
    Cached per entity, so an interrupted extraction resumes."""
    os.makedirs(cache_dir, exist_ok=True)
    system = EXTRACT_SYSTEM.replace("<<RULES>>", src.rules)
    facts, cost, streams = {}, dict(in_tokens=0, out_tokens=0, seconds=0.0, llm_calls=0), {}
    for e, g in src.by_entity.items():
        p = f"{cache_dir}/{_safe(e)}.csv"
        if os.path.exists(p) and S.llm is not None:
            facts[e] = pd.read_csv(p, keep_default_na=False, dtype=str).to_dict("records")
            for x in facts[e]:
                x["subject"] = canon_subject(x["subject"], str(e), x.get("attribute", ""))
            normalize_facts(facts[e], src.name, str(e))
            c = json.load(open(p + ".cost.json"))
            for k in cost:
                cost[k] += c.get(k, 0)
        else:
            facts[e] = []
            streams[e] = list(g.itertuples())

    def make_conv(e, r):
        return [{"role": "system", "content": system},
                {"role": "user", "content": f"Facts already in the graph for {r.entity}:\n{state_lines(facts[e])}\n\n"
                                            f"New evidence (published {r.known_at:%Y-%m-%d %H:%M:%S}):\n[{r.evidence_id}] {r.text}"}]

    ecost = {e: dict(in_tokens=0, out_tokens=0, seconds=0.0, llm_calls=0) for e in streams}

    def update(e, r, o):
        for it in _parse_facts(o["text"]):
            facts[e].append(dict(evidence_id=r.evidence_id, learned_at=f"{r.known_at:%Y-%m-%d %H:%M:%S}",
                                 subject=canon_subject(it.get("subject", ""), str(r.entity), it.get("attribute", "")),
                                 attribute=str(it.get("attribute", "")).strip(),
                                 period=str(it.get("period") or ""), value=str(it.get("value", "")),
                                 effective_at=str(it.get("effective_at") or ""), kind=str(it.get("kind", ""))))
        c = ecost[e]
        c["in_tokens"] += o["in_tokens"]; c["out_tokens"] += o["out_tokens"]; c["seconds"] += o["seconds"]; c["llm_calls"] += 1
        if c["llm_calls"] == len(streams[e]) and S.llm is not None:      # entity finished -> checkpoint
            pd.DataFrame(facts[e], columns=FACT_COLS).to_csv(f"{cache_dir}/{_safe(e)}.csv", index=False)
            json.dump(c, open(f"{cache_dir}/{_safe(e)}.csv.cost.json", "w"))

    if streams:
        print(f"  [extract] {src.name}: {sum(len(v) for v in streams.values())} evidence lines, "
              f"{len(streams)} entities (batched across entities)", flush=True)
        c = lockstep(streams, make_conv, update, max_tokens=1500, label=f"extract {src.name}")
        for k in cost:
            cost[k] += c[k]
    for e in src.by_entity:
        normalize_facts(facts[e], src.name, str(e))
    rows = [x for e in src.by_entity for x in facts[e]]
    return pd.DataFrame(rows, columns=FACT_COLS), cost


def _F(r, subject, attribute, period, value, effective_at, kind):
    return dict(evidence_id=r.evidence_id, learned_at=f"{r.known_at:%Y-%m-%d %H:%M:%S}", subject=subject,
                attribute=attribute, period=period, value=f"{value:g}" if isinstance(value, float) else str(value),
                effective_at="" if effective_at == "" else f"{pd.Timestamp(effective_at):%Y-%m-%d %H:%M:%S}", kind=kind)


def extract_oracle(src):
    """Perfect facts from the gold columns (upper bound). Per entity: brokers, splits and renames are per ticker."""
    facts = []
    for e, g in src.by_entity.items():
        alias, splits, last, seen = {}, [], {}, {}
        for r in g.itertuples():
            if src.name == "broker":
                if r.gold_event == "split":
                    splits.append((r.known_at, float(r.gold_split_ratio)))
                    facts.append(_F(r, r.entity, "split_ratio", "", float(r.gold_split_ratio), r.known_at, "event")); continue
                if r.gold_event == "rename":
                    alias[r.gold_new_name] = alias.get(r.gold_firm, r.gold_firm)
                    facts.append(_F(r, f"{r.entity} | {r.gold_firm}", "alias", "", r.gold_new_name, r.known_at, "event")); continue
                firm = alias.get(r.gold_firm, r.gold_firm); subj = f"{r.entity} | {firm}"
                adj = lambda x, t: x / float(np.prod([k for s, k in splits if t < s] or [1.0]))
                moved = r.gold_pt_action in ("Raises", "Lowers") and pd.notna(r.gold_prior_pt)
                if firm in last and moved and pd.notna(last[firm][1]):
                    prev = adj(last[firm][1], last[firm][0])
                    if abs(prev - r.gold_prior_pt) > max(1e-9, 0.005 * abs(prev)):     # missed revision (0.5% rounding)
                        facts.append(_F(r, subj, "price_target_usd", "", float(r.gold_prior_pt),
                                        last[firm][0] + pd.Timedelta(seconds=1), "inferred"))
                changed = firm not in last or r.gold_pt_action in ("Raises", "Lowers", "Announces")
                if pd.notna(r.gold_current_pt):
                    facts.append(_F(r, subj, "price_target_usd", "", float(r.gold_current_pt), r.known_at,
                                    "new" if changed else "corroborate"))
                if isinstance(r.gold_to_grade, str) and r.gold_to_grade:
                    facts.append(_F(r, subj, "rating", "", r.gold_to_grade, r.known_at, "new" if changed else "corroborate"))
                last[firm] = (r.known_at, r.gold_current_pt if pd.notna(r.gold_current_pt) else (last.get(firm, (0, np.nan))[1]))
            elif src.name == "alfred":
                label = getattr(r, "gold_label", "") or r.gold_series
                label = label if isinstance(label, str) and label else "real GDP growth % SAAR"
                key = r.gold_period
                facts.append(_F(r, f"{r.entity} | {label}", "estimate", str(r.gold_period), float(r.gold_value), "",
                                "new" if key not in seen else "revision"))
                seen[key] = True
            elif src.name == "edgar":
                per = f"{r.gold_start}..{r.gold_end}"; v = float(r.gold_value_musd)
                kind = "new" if per not in seen else ("corroborate" if seen[per] == v else "revision"); seen[per] = v
                facts.append(_F(r, f"{r.entity} | {r.gold_concept}", "value_usd_m", per, v, "", kind))
    return pd.DataFrame(facts, columns=FACT_COLS), dict(in_tokens=0, out_tokens=0, seconds=0.0, llm_calls=0)


def build_store(src, extraction):
    """Fact store of one source. LLM extraction is cached per model (shared by methods 3 and 4)."""
    if extraction == "oracle":
        f, cost = extract_oracle(src)
    else:
        f, cost = extract_llm(src, os.path.join(S.args.out_dir, S.model, "cache", f"facts_{src.name}_llm"))
    f = f.fillna("").astype(str).copy()
    f.insert(0, "fact_id", range(1, len(f) + 1))
    return f, cost


def entity_facts(f, entity):
    return f[(f.subject == entity) | f.subject.str.startswith(entity + " |")]


# ====================================================================================== summaries: accuracy, cost, time
def _load_results(out_dir, method=None):
    pat = f"{out_dir}/*/{method or '*'}/*.csv"
    files = [p for p in glob.glob(pat) if "/cache/" not in p]
    if not files:
        return None, {}
    res = pd.concat([pd.read_csv(p, keep_default_na=False, dtype={"gold": str, "answer": str}) for p in files], ignore_index=True)
    for c in ("correct", "cited_future"):
        res[c] = res[c].astype(str).str.lower().eq("true")
    for c in ("in_tokens", "out_tokens", "seconds"):
        res[c] = pd.to_numeric(res[c], errors="coerce").fillna(0)
    res["n_dropped"] = pd.to_numeric(res.get("n_dropped", 0), errors="coerce").fillna(0)
    metas = {}
    for p in glob.glob(f"{out_dir}/*/{method or '*'}/*_meta.json"):
        m = json.load(open(p)); metas[(m["model"], m["method"], m["variant"])] = m
    return res, metas


def summarize(out_dir, method=None, quiet=False):
    """accuracy / cost / time per model x method x variant x source (+ ALL). Writes summary_<method or all>.csv."""
    res, metas = _load_results(out_dir, method)
    if res is None:
        print("no results yet"); return None
    res["peek"] = (res.category == "peek") | res.cited_future
    res["base_variant"] = res.variant.str.replace(r"_r\d+$", "", regex=True)
    rows = []
    keys = ["model", "method", "base_variant"]
    for k, g in res.groupby(keys):
        variants = g.variant.unique()
        for source, h in [("ALL", g)] + list(g.groupby("source")):
            ing_in = ing_out = ing_sec = calls = 0.0; wall = 0.0
            for v in variants:
                m = metas.get((k[0], k[1], v), {})
                wall += m.get("wall_seconds_answering", 0) if source == "ALL" else 0
                for s, c in (m.get("ingestion") or {}).items():
                    if source in ("ALL", s):
                        ing_in += c.get("in_tokens", 0); ing_out += c.get("out_tokens", 0)
                        ing_sec += c.get("seconds", 0); calls += c.get("llm_calls", 0)
            asof, today = h[h["mode"] == "as_of"], h[h["mode"] == "today"]
            lat = [x for v in variants for x in metas.get((k[0], k[1], v), {}).get("latency_seconds", [])]
            m0 = metas.get((k[0], k[1], variants[0]), {})
            rows.append(dict(
                backend=m0.get("backend", ""), gpu=m0.get("gpu", ""),
                model=k[0], method=k[1], variant=k[2], source=source, runs=len(variants), n=len(h),
                # --- accuracy
                accuracy=h.correct.mean(), acc_today=today.correct.mean() if len(today) else np.nan,
                acc_as_of=asof.correct.mean() if len(asof) else np.nan,
                peek_rate_as_of=asof.peek.mean() if len(asof) else np.nan,
                trap_rate=(h.category == "trap").mean(), false_abstain_rate=(h.category == "false_abstain").mean(),
                wrong_rate=(h.category == "wrong").mean(),
                truncated_rate=(h.n_dropped > 0).mean(),
                # --- cost (tokens)
                in_tokens_per_q=h.in_tokens.mean(), out_tokens_per_q=h.out_tokens.mean(),
                answer_tokens_total=int(h.in_tokens.sum() + h.out_tokens.sum()),
                ingestion_tokens_total=int(ing_in + ing_out), ingestion_llm_calls=int(calls),
                total_tokens=int(h.in_tokens.sum() + h.out_tokens.sum() + ing_in + ing_out),
                tokens_per_correct=(h.in_tokens.sum() + h.out_tokens.sum() + ing_in + ing_out) / max(1, h.correct.sum()),
                # --- time (seconds)
                sec_per_q=h.seconds.mean(), answer_seconds_total=h.seconds.sum(), ingestion_seconds=ing_sec,
                total_seconds=h.seconds.sum() + ing_sec,
                answering_wall_seconds=wall if source == "ALL" else np.nan,
                latency_p50_s=float(np.median(lat)) if lat and source == "ALL" else np.nan,
                latency_mean_s=float(np.mean(lat)) if lat and source == "ALL" else np.nan))
    out = pd.DataFrame(rows)
    stab = res.groupby(keys + ["qid"]).answer.agg(lambda a: a.map(norm).nunique() == 1).groupby(keys).mean()
    out["stability"] = [stab.get((r.model, r.method, r.variant), np.nan) if r.runs > 1 and r.source == "ALL" else np.nan
                        for r in out.itertuples()]
    out = out.round(4)
    name = f"summary_{method}" if method else "summary_all_methods"
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(f"{out_dir}/{name}.csv", index=False)
    if not quiet:
        show = out[out.source == "ALL"][["model", "method", "variant", "n", "accuracy", "acc_today", "acc_as_of",
                                         "peek_rate_as_of", "trap_rate", "in_tokens_per_q", "out_tokens_per_q",
                                         "ingestion_tokens_total", "sec_per_q", "latency_p50_s", "ingestion_seconds", "truncated_rate"]]
        with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_columns", 30):
            print(f"\n=== {name}: accuracy | cost (tokens) | time (s)  [per-source rows in the CSV] ===")
            print(show.to_string(index=False))
        print(f"written: {out_dir}/{name}.csv")
    return out


# ====================================================================================== driver
def main(method, add_args, run_method, argv=None, description=""):
    p = common_args(description)
    add_args(p)
    S.args = a = p.parse_args(argv)
    S.seed = 0
    S.backend = resolve_backend(a)
    for k in ("out_dir", "responses_dir", "data_dir"):
        setattr(a, k, os.path.abspath(getattr(a, k)))
    # results of different engines are never mixed (timing is only comparable within one engine)
    a.out_dir, a.responses_dir = os.path.join(a.out_dir, S.backend), os.path.join(a.responses_dir, S.backend)
    print(f"[engine] backend={S.backend}  results -> {a.out_dir}", flush=True)
    sources = None
    for key in (["dry_run"] if a.dry_run else a.models):
        S.model = key
        S.llm = None if a.dry_run else make_engine(key, a)
        try:
            sources = load_sources()       # token counts depend on the tokenizer, so reload per model
            if not sources:
                sys.exit("no questions selected")
            for r in range(a.runs):
                S.seed = r
                run_method(sources)
        finally:
            if S.llm is not None:
                S.llm.close(); S.llm = None
    summarize(a.out_dir, method)