# -*- coding: utf-8 -*-
"""
Compute Hessian wrt PARAMETERS (θ) for EleutherAI/pythia-* checkpoints.
- Uses PyHessian (HVP-based) to avoid materializing the full Hessian.
- Measures: top eigenvalues, trace(H), ||H||_F^2, stable rank.
- Uses a single last-position CE loss on the prompt (teacher forcing).
"""

import os
import re
import csv
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    from torch.backends.cuda import sdp_kernel
    sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import list_repo_refs

# --------- You may need: pip install pyhessian ---------
from pyhessian import hessian as pyhessian_hessian

# --- Optional shim for older pyhessian that calls torch.eig ---
def _torch_eig_wrapper(*args, eigenvectors=True, **kwargs):
    # torch.linalg.eig returns complex eigenvalues & eigenvectors
    evals, evecs = torch.linalg.eig(*args, **kwargs)
    evals_real = evals.real
    evals_imag = evals.imag
    evals_combined = torch.stack((evals_real, evals_imag), dim=-1)
    return evals_combined, evecs
if not hasattr(torch, "eig"):
    torch.eig = _torch_eig_wrapper  # type: ignore

# ---------------- Config ----------------
MODEL_ID  = "EleutherAI/pythia-70m-deduped"
PROMPT    = "The quick brown fox jumps over the lazy dog"
DEVICE    = "cuda" #if torch.cuda.is_available() else "cpu"
HF_HOME   = os.getenv("HF_HOME", None)

# Scan many revisions? keep it modest the first time.
MAX_STEPS   = None        # e.g., 50 to limit; None = all
STEP_STRIDE = 1           # e.g., 10 to sample every 10th step

TOP_N_EIG   = 1           # top-k eigenvalues from PyHessian
SEED        = 1234        # for reproducibility of Hutchinson-type estimates (trace)

torch.manual_seed(SEED)

@dataclass
class ParamMetrics:
    revision: str
    step: int
    loss: float
    max_eig: float
    fro_sq: float
    trace: float
    stable_rank: float
    num_params: int


def get_step_tags(model_id: str):
    refs = list_repo_refs(model_id)
    step_refs = []
    for ref in list(refs.tags) + list(refs.branches):
        m = re.fullmatch(r"step(\d+)", ref.name)
        if m:
            step_refs.append((ref.name, int(m.group(1))))
    step_refs.sort(key=lambda x: x[1])
    return step_refs


@torch.no_grad()
def _tokenize(tokenizer, text, device):
    return tokenizer(text, return_tensors="pt").to(device)


# ---- A tiny wrapper so PyHessian can call model(inputs) -> logits ----
class LMHeadWrapper(nn.Module):
    """
    Forward returns logits for the given (input_ids, attention_mask).
    The criterion will decide how to turn logits -> loss.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        input_ids, attention_mask = inputs
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits  # (batch size B, sequence length T, vocabulary size V)


def last_position_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy on the LAST position only.
    logits: (B, T, V)
    target: (B,) — the gold id for the last position
    """
    last_logits = logits[:, -1, :]               # (B, V)
    return F.cross_entropy(last_logits, target)


def _param_hessian_for_revision(model_id: str, revision: str, tokenizer, text: str) -> ParamMetrics:
    # 1) Load model @ specific revision
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.float32,      # keep autograd simple & stable
        low_cpu_mem_usage=True,
        cache_dir=HF_HOME,
        attn_implementation="eager",
    )
    base_model.eval()
    
    num_params_total = sum(p.numel() for p in base_model.parameters())
    num_params_trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    # (The Hessian lives in R^{N x N} with N = num_params_trainable)
    print(f"Params: total={num_params_total:,}  trainable={num_params_trainable:,}")

    # Device selection: PyHessian can run on CPU (slow) or CUDA.
    use_cuda = (torch.cuda.is_available() and DEVICE == "cuda")
    if use_cuda:
        from torch.backends.cuda import sdp_kernel
        # Disable flash + mem-efficient; force math kernel (supports higher-order grads)
        sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)

    # 2) Prepare a single (context, label) batch: predict the last token
    with torch.no_grad():
        batch = _tokenize(tokenizer, text, DEVICE if use_cuda else "cpu")
        input_ids = batch["input_ids"]         # (1, T)
        attn_mask = batch.get("attention_mask", torch.ones_like(input_ids))

        # context (up to T-1), label (the T-th / last token)
        ctx_ids   = input_ids[:, :-1]
        ctx_mask  = attn_mask[:, :-1]
        target    = input_ids[:, -1]           # (1,)

        # sanity: require length ≥ 2
        assert ctx_ids.shape[1] >= 1, "Need at least 2 tokens in PROMPT."

    # 3) Wrap for PyHessian
    wrapped = LMHeadWrapper(base_model)

    # PyHessian API expects data=(inputs, targets); inputs can be a tuple
    # Here: inputs=(ctx_ids, ctx_mask), target=target
    # NOTE: Setting cuda=use_cuda toggles GPU usage inside PyHessian’s HVPs.
    H = pyhessian_hessian(
        wrapped,
        last_position_ce,
        data=((ctx_ids, ctx_mask), target),
        cuda=use_cuda
    )

    # 4) Compute stats
    #    - eigenvalues: uses Lanczos/power iteration on HVPs
    top_evals = H.eigenvalues(top_n=TOP_N_EIG)[0]   # -> list/np.array
    max_eval  = float(top_evals[0])

    #    - trace(H)
    tr_est, _ = H.trace()          # not float(H.trace())
    tr = float(tr_est)


    with torch.no_grad():
        logits_log = wrapped((ctx_ids.to(base_model.device), ctx_mask.to(base_model.device)))
        loss_log = last_position_ce(logits_log, target.to(base_model.device)).item()
        
    # Estimate Frobenius norm squared via Hutchinson’s method
    def loss_closure():
        logits = wrapped((ctx_ids.to(base_model.device), ctx_mask.to(base_model.device)))
        return last_position_ce(logits, target.to(base_model.device))  # Tensor, no .item()

    loss_for_hutch = loss_closure()
    params = [p for p in base_model.parameters() if p.requires_grad]
    fro_sq_est = frobenius_sq_hutchinson(loss_for_hutch, params, num_samples=32)
    stable_rank = fro_sq_est / (max_eval**2 + 1e-12)

    # Count parameters
    num_params = sum(p.numel() for p in base_model.parameters())

    # Parse step
    m = re.fullmatch(r"step(\d+)", revision)
    step = int(m.group(1)) if m else -1

    # Cleanup
    del H, wrapped, base_model, logits
    if use_cuda:
        torch.cuda.empty_cache()

    return ParamMetrics(
        revision=revision,
        step=step,
        loss=loss,
        max_eig=max_eval,
        fro_sq=fro_sq_est,
        trace=tr,
        stable_rank=stable_rank,
        num_params=num_params
    )

# --- Hutchinson approximation of Frobenius norm (||H||_F^2) ---

def _make_like_params(params):
    """Create a list of random Rademacher vectors (+1/-1) matching each param's shape."""
    vecs = []
    for p in params:
        if not p.requires_grad:
            vecs.append(None)
        else:
            v = torch.empty_like(p).bernoulli_(0.5).mul_(2).sub_(1)
            vecs.append(v)
    return vecs

def _list_sqnorm(tensors):
    """Compute squared L2 norm across a list of tensors."""
    s = 0.0
    for t in tensors:
        if t is not None:
            s += (t.float() ** 2).sum().item()
    return s

def hvp(loss, params, vec):
    """Compute Hessian–vector product H v for scalar loss."""
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused=True)
    dot = 0.0
    for g, v in zip(grads, vec):
        if g is not None and v is not None:
            dot = dot + (g * v).sum()
    hv = torch.autograd.grad(dot, params, retain_graph=True, allow_unused=True)
    return hv

def frobenius_sq_hutchinson(loss, params, num_samples=32):
    """Estimate ||H||_F^2 via Hutchinson’s trace estimator."""
    est = 0.0
    for _ in range(num_samples):
        v = _make_like_params(params)
        Hv = hvp(loss, params, v)
        est += _list_sqnorm(Hv)
    return est / num_samples

def main():
    print(f"Using {DEVICE} | Model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=HF_HOME)

    step_tags = get_step_tags(MODEL_ID)
    if not step_tags:
        print("No step tags found; analyzing default branch only.")
        step_tags = [("main", 0)]

    # stride & cap
    step_tags = step_tags[::STEP_STRIDE]
    if MAX_STEPS is not None:
        step_tags = step_tags[:MAX_STEPS]

    results = []
    for rev, step in step_tags:
        print(f"==> {rev} (step {step})")
        try:
            m = _param_hessian_for_revision(MODEL_ID, rev, tokenizer, PROMPT)
        except RuntimeError as e:
            print(f"  Skipped {rev}: {e}")
            continue
        results.append(m)
        print(f"  loss={m.loss:.4f}  max_eig={m.max_eig:.6g}   trace={m.trace:.6g}")

    # Save CSV
    out_csv = "pythia_param_hessian.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["revision", "step", "loss", "max_eig", "fro_sq", "trace", "stable_rank", "num_params"])
        for m in results:
            w.writerow([m.revision, m.step, m.loss, m.max_eig, m.trace, m.num_params])

    print(f"\nSaved {len(results)} rows to {out_csv}")


if __name__ == "__main__":
    main()