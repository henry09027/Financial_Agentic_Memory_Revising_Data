#!/usr/bin/env python3
"""Pre-flight check for a new session: imports that failed before, GPU/driver, model support, disk.
Prints a fix for every known failure. No model weights are downloaded (only small config files)."""
import os, shutil, sys

ok = True
FIX = {
    "GLIBCXX": "conda base's pyzmq is on the path. Use the venv (source env.sh), or: pip install --user --force-reinstall --no-deps pyzmq",
    "libnvrtc.so.13": "torchcodec is a CUDA 13 build. Run: pip install --no-deps --force-reinstall 'torchcodec==0.16.*' --index-url https://download.pytorch.org/whl/cpu",
    "driver on your system is too old": "this torch/vLLM build needs a newer driver. Use the pinned stack: bash setup_env.sh (vLLM 0.32.0 + CUDA 12.9)",
    "Disk quota": "home quota full. source env.sh so caches/downloads go to the work disk, and free ~/.cache (see README step 0)",
}


def fail(what, e):
    global ok
    ok = False
    msg = str(e)
    print(f"  FAIL {what}: {msg[:200]}")
    for k, v in FIX.items():
        if k in msg:
            print(f"       fix: {v}")


print(f"python {sys.version.split()[0]}  ({sys.executable})")
print(f"venv active: {sys.prefix != sys.base_prefix}   user site ignored: {not __import__('site').ENABLE_USER_SITE}")
hf = os.environ.get("HF_HOME")
print(f"HF_HOME = {hf or '(not set: downloads would go to ~/.cache/huggingface -> source env.sh)'}")
if hf:
    os.makedirs(hf, exist_ok=True)
    print(f"  free on HF_HOME disk: {shutil.disk_usage(hf).free / 1e9:.0f} GB (the three models need ~33 GB)")
if not hf:
    ok = False

import tempfile
t = tempfile.gettempdir()
print(f"temp dir: {t}  ({'ok' if len(t) <= 60 else 'too long for vLLM IPC sockets; scripts will use /tmp. Remove TMPDIR from env.sh'})")
print("imports:")
for mod in ("numpy", "pandas", "sklearn", "requests", "yfinance", "zmq", "torch", "torchcodec", "transformers", "vllm"):
    try:
        m = __import__(mod)
        print(f"  ok   {mod:13} {getattr(m, '__version__', '')}")
    except Exception as e:
        fail(mod, e)

try:
    import torch
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}  | torch built for CUDA {torch.version.cuda}")
    torch.zeros(1, device="cuda")      # is_available() only sees the GPU; a real allocation checks driver vs build
    print("  ok   CUDA init")
except Exception as e:
    fail("CUDA init", e)

try:
    from vllm import LLM  # noqa: F401  (pulls in torchcodec, zmq, flashinfer: the imports that broke before)
    print("  ok   from vllm import LLM")
except Exception as e:
    fail("from vllm import LLM", e)

supported = None
try:
    from vllm import ModelRegistry
    supported = set(ModelRegistry.get_supported_archs())
except Exception as e:
    print(f"  (could not read vLLM model registry: {str(e)[:120]})")
try:
    from transformers import AutoConfig
    for hf_id in ("microsoft/Phi-4-mini-instruct", "Qwen/Qwen3.5-4B", "google/gemma-4-E4B-it"):
        try:
            archs = AutoConfig.from_pretrained(hf_id).architectures or []
            sup = "?" if supported is None else all(a in supported for a in archs)
            print(f"  {'ok  ' if sup is not False else 'FAIL'} {hf_id:32} {archs}  vllm supports: {sup}")
            ok &= sup is not False
        except Exception as e:
            fail(hf_id, e)
except Exception as e:
    fail("transformers AutoConfig", e)

print("\nOK: ready to run" if ok else "\nPROBLEM: see FAIL lines above")
sys.exit(0 if ok else 1)
