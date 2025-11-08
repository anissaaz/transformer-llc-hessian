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

from collections import UserDict
from collections.abc import MutableMapping

import torch.utils.data as data_utils
from torch import Tensor, bfloat16, eye, manual_seed, no_grad
from torch.nn import CrossEntropyLoss, Module
from transformers import (
    DataCollatorWithPadding,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)

from curvlinops import HessianLinearOperator
from curvlinops import hutchinson_trace
from curvlinops import hutchinson_squared_fro

# make deterministic
manual_seed(0)

# %%
#
# Data
# ----
#
# We will use synthetic data for simplicity. But obviously this can
# be replaced with any HF dataloader.

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id

texts = [
    "The mouse ran away from the cat"
]

batch = tokenizer(texts, return_tensors="pt")
# Teacher forcing labels = input_ids (no padding here, so no masking needed)
batch["labels"] = batch["input_ids"][:,1:].clone()
batch["input_ids"] = batch["input_ids"][:,:-1].clone()

# import ipdb; ipdb.set_trace()

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


class MyGPT2(Module):
    """
    Huggingface LLM wrapper.

    Args:
        tokenizer: The tokenizer used for preprocessing the text data. Needed
            since the model needs to know the padding token id.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()
        self.hf_model = AutoModelForCausalLM.from_pretrained("gpt2")

        # For simplicity, only enable grad for the last layer
        for p in self.hf_model.parameters():
            p.requires_grad = False

        for p in self.hf_model.lm_head.parameters():
            p.requires_grad = True

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
        return logits.reshape(logits.shape[0] * logits.shape[1], -1)                                           # (batch_size, sequence_length, vocabulary_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyGPT2(tokenizer).to(device=device, dtype=bfloat16)


with no_grad():
    logits = model(batch)
    print(f"Logits shape: {logits.shape}")


# %%
#
# Curvlinops
# ----------
#
# We are now ready to compute the curvature of this HF model using Curvlinops.
# For this, we need to define a function to tell Curvlinops how to get the
# batch size of the :code:`UserDict` input batch. Everything else is unchanged
# from the standard usage of Curvlinops!


def batch_size_fn(x: MutableMapping):
    return x["input_ids"].shape[0]


params = [p for p in model.parameters() if p.requires_grad]

def ce_loss(logits, labels):
    logits = logits[:, :-1, :]                      # predict token t using context up to t-1
    labels = labels[:, 1:]                          # next-token target
    import ipdb; ipdb.set_trace()
    return F.cross_entropy(logits, labels)

 
    

hessian = HessianLinearOperator(
	model,
	CrossEntropyLoss(),
	params,
	[(batch, batch["labels"].flatten())],
	check_deterministic=False,              # don't check randomness
	batch_size_fn=batch_size_fn,
)

#H = hessian @ eye(hessian.shape[0], device=params[0].device)

print(f"Hessian shape: {hessian.shape}")


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
    print(f"Leading {k} Hessian eigenvalues: {evals}")
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
    trace_estimate = hutchinson_trace(H, num_matvecs)
    print(f"Trace estimate: {trace_estimate}")
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
    print(f"Stable rank: {stable_rank}")
    return stable_rank

top_k_evals(hessian, 5)
hutchinson_trace_estimate(hessian)
stable_rank(hessian)