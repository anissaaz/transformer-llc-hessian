r"""Usage with Huggingface LLMs
===============================

This example demonstrates how to work with Huggingface (HF) language models.

As always, let's first import the required functionality.
Remember to run :code:`pip install -U transformers datasets`
"""

import scipy
import scipy.sparse.linalg
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import re
import os

from collections import UserDict
from collections.abc import MutableMapping

from dataclasses import dataclass, asdict
from datasets import load_dataset

import torch.utils.data as data_utils
from torch import Tensor, bfloat16, eye, manual_seed, no_grad
from torch.nn import CrossEntropyLoss, Module
from transformers import (
    DataCollatorWithPadding,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)

from huggingface_hub import list_repo_refs
from huggingface_hub import hf_hub_download

from curvlinops import HessianLinearOperator
from curvlinops import hutchinson_trace
from curvlinops import hutchinson_squared_fro

try:
    # PyTorch 2.x: global setting for scaled-dot-product attention kernels
    from torch.backends.cuda import sdp_kernel
    sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
except Exception:
    # Fallback: older versions just won't have flash SDPA anyway
    pass


@dataclass
class HessianMetrics:
    revision: str
    step: int
    trace: float
    max_eig: float
    stable_rank: float

# make deterministic
manual_seed(0)

# %%
#
# Data
# ----
#
# We will use synthetic data for simplicity. But obviously this can
# be replaced with any HF dataloader.

# --------- Config ---------
#MODEL = "EleutherAI/pythia-70m-deduped"
EXPERIMENT_DIR = "hessian-batch40-47"
os.makedirs(EXPERIMENT_DIR, exist_ok=True)
MAX_LEN = 256

def get_step_tags(model_id: str):
    refs = list_repo_refs(model_id)
    step_refs = []
    for ref in list(refs.tags) + list(refs.branches):
        m = re.fullmatch(r"step(\d+)", ref.name)
        if m:
            step_refs.append((ref.name, int(m.group(1))))
    step_refs.sort(key=lambda x: x[1])
    return step_refs

# %%
#
# Model
# -----
#
# Curvlinops supports general :code:`UserDict` inputs. However, everything must
# be handled inside the :code:`forward` function of the model. This gives
# the users the most flexibility, without much overhead.
#
# Let's wrap the HF model to conform this requirement then.


class MyTransformer(Module):
    """
    Huggingface LLM wrapper.

    Args:
        tokenizer: The tokenizer used for preprocessing the text data. Needed
            since the model needs to know the padding token id.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, model, revision) -> None:
        super().__init__()
        self.hf_model = AutoModelForCausalLM.from_pretrained(model, revision=revision)

        # grad only for last layer
        # for p in self.hf_model.parameters():
        #     p.requires_grad = False
            
        # head = self.hf_model.get_output_embeddings()
        
        # for p in head.parameters():
        #     p.requires_grad = True

    def forward(self, data: MutableMapping) -> Tensor:
        """
        Custom forward function. Handles things like moving the
        input tensor to the correct device inside.

        Args:
            data: A dict-like data structure with `input_ids` inside.
                This is the default data structure assumed by Huggingface
                dataloaders.

        Returns:
            logits: An `(batch_size, sequence_length, vocab_size)`-sized tensor of logits.
        """
        device = next(self.parameters()).device
        input_ids = data["input_ids"].to(device)
        attention = data["attention_mask"].to(device)
        outputs = self.hf_model(input_ids, attention_mask=attention)
        logits = outputs.logits
        return logits.reshape(logits.shape[0] * logits.shape[1], -1)

# with no_grad():
#     logits = model(batch)
#     print(f"Logits shape: {logits.shape}")

def batch_size_fn(x: MutableMapping):
    return x["input_ids"].shape[0]

# %%
#
# Curvlinops
# ----------
#
# We are now ready to compute the curvature of this HF model using Curvlinops.
# For this, we need to define a function to tell Curvlinops how to get the
# batch size of the :code:`UserDict` input batch. Everything else is unchanged
# from the standard usage of Curvlinops!





# def ce_loss(logits, labels):
#     return F.cross_entropy(logits, labels, reduction=ce_loss.reduction)

# ce_loss.reduction = 'mean'

# import ipdb; ipdb.set_trace()

#H = hessian @ eye(hessian.shape[0], device=params[0].device)


def top_k_evals(H, k=3, which="LM"):
    """
    Compute the top-k eigenvalues of a Curvlinops linear operator.

    Args:
        H: A Curvlinops linear operator (e.g., HessianLinearOperator)
        k: Number of eigenvalues to compute (default 3)
        which: Which eigenvalues to compute ('LM' = largest magnitude)

    Returns:
        A NumPy array of the top-k eigenvalues.
    """
    # Convert to a SciPy-compatible LinearOperator
    H_scipy = H.to_scipy()

    # Compute eigenvalues
    evals, _ = scipy.sparse.linalg.eigsh(H_scipy, k=k, which=which)
    evals = np.round(evals, 3)
    # print(f"Leading {k} Hessian eigenvalues: {evals}")
    return evals

def hutchinson_trace_estimate(H, num_matvecs=5):
    """
    Estimate the trace of a linear operator (e.g. Hessian) using Hutchinson’s method.

    Args:
        H: A Curvlinops linear operator or matrix (e.g. HessianLinearOperator).
        num_matvecs: Number of random vectors to use in the Monte Carlo estimate.
                     Higher values give more accurate estimates (default: 5).
        distribution: Random vector distribution ('rademacher' or 'normal').
                      Default is Rademacher (+1/-1 entries).

    Returns:
        trace_estimate: Estimated trace of H as a scalar (PyTorch tensor or float).
    """
    trace_estimate = hutchinson_trace(H, num_matvecs).item()
    # print(f"Trace estimate: {trace_estimate}")
    return trace_estimate

def stable_rank(H, num_matvecs=5):
    """
    Estimate stable rank(H) = ||H||_F^2 / (lambda_max(H))^2.
    - H: Curvlinops linear operator (e.g., HessianLinearOperator)
    - num_matvecs: Hutchinson samples for Frobenius^2 (more = less variance)
    - which: 'LA' (largest algebraic) or 'LM' (largest magnitude) for eigsh
    """
    # Frobenius norm squared via Hutchinson
    fro2 = hutchinson_squared_fro(H, num_matvecs).item()
    
    # Top eigenvalue (scalar)
    max_eig = float(top_k_evals(H, k=1)[0])
    
    stable_rank = fro2 / (max_eig**2)
    stable_rank = round(stable_rank, 3)
    return stable_rank

def run_hessian_analysis(model_name, part=None, output_suffix="full_model"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    step_tags = get_step_tags(model_name)
    step_tags = [
        (rev, step)
        for rev, step in step_tags
        if step in {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 140000}
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    dataset = load_dataset("EleutherAI/the_pile_deduplicated", split ="train[:1%]")
    subset = dataset.shuffle(seed=0).select(range(40, 48))
    texts = [tokenizer.bos_token + ex["text"] + tokenizer.eos_token for ex in subset]

    batch = tokenizer(
        texts, 
        return_tensors="pt",
        padding = True,
        truncation = True,
        max_length=MAX_LEN,
        )
    batch["labels"] = batch["input_ids"][:,1:].clone()
    batch["input_ids"] = batch["input_ids"][:,:-1].clone()
    batch["attention_mask"] = batch["attention_mask"][:,:-1].clone()        # align with input_ids ?
    
    batch["labels"][batch["attention_mask"] == 0] = -100       # labels adjusted for ignore_index=-100 for CE loss
    
    #import ipdb; ipdb.set_trace()
    
    results: list[HessianMetrics] = []
    printed_shape = False
    
    for rev, step in step_tags:
        print(f"==> {rev}")
        model = MyTransformer(tokenizer, model_name, revision=rev).to(device=device, dtype=bfloat16)
        
        #import ipdb; ipdb.set_trace()
        if part is not None:
            params = [
                tensor                                          # store only the tensor
                for name, tensor in model.named_parameters() 
                if part in name
                ]
        else:
            params = [
                tensor
                for name, tensor in model.named_parameters()
            ]
        
        hessian = HessianLinearOperator(
            model,
            CrossEntropyLoss(),
            params,
            [(batch, batch["labels"].flatten())],
            check_deterministic=False,              # don't check randomness
            batch_size_fn=batch_size_fn,
        )
        
        # print the Hessian shape only once
        if not printed_shape:
            print(f"Hessian shape: {hessian.shape}")
            printed_shape = True
        
        # Compute metrics
        max_eig = top_k_evals(hessian, k=1)[0]
        trace_val = hutchinson_trace_estimate(hessian, num_matvecs=5)       # returns torch.Tensor
        stable_rank_val = stable_rank(hessian, num_matvecs=5)
        
        print(f"max eig = {max_eig:.3f}, trace = {trace_val:.3f}, stable rank = {stable_rank_val:.3f}")
        
        results.append(HessianMetrics(
            revision=rev,
            step=step,
            trace=trace_val,
            max_eig=max_eig,
            stable_rank=stable_rank_val,
        ))
    
    csv_name = os.path.join(EXPERIMENT_DIR, f"hessian_metrics_{output_suffix}_40-47.csv")
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(csv_name, index=False)
    print(f"\nSaved results to {csv_name}")
    
if __name__ == "__main__":
    run_hessian_analysis("EleutherAI/pythia-14m")