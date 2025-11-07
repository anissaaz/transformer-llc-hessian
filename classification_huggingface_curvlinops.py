r"""Usage with Huggingface LLMs
===============================

This example demonstrates how to work with Huggingface (HF) language models.

As always, let's first import the required functionality.
Remember to run :code:`pip install -U transformers datasets`
"""

import scipy
import scipy.sparse.linalg
import torch
import numpy as np

from collections import UserDict
from collections.abc import MutableMapping

import torch.utils.data as data_utils
from datasets import Dataset
from torch import Tensor, bfloat16, eye, manual_seed, no_grad
from torch.nn import CrossEntropyLoss, Module
from transformers import (
    DataCollatorWithPadding,
    GPT2Config,
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
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

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id

data = [
    {"text": "Today is hot, but I will manage!!!!", "label": 1},
    {"text": "Tomorrow is cold", "label": 0},
    {"text": "Carpe diem", "label": 1},
    {"text": "Tempus fugit", "label": 1},
]
dataset = Dataset.from_list(data)


def tokenize(row):
    return tokenizer(row["text"])


dataset = dataset.map(tokenize, remove_columns=["text"])
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
dataloader = data_utils.DataLoader(
    dataset, batch_size=100, collate_fn=DataCollatorWithPadding(tokenizer)
)

# %%
#
# Let's check the batch emitted by HF. We will see that it is a :code:`UserDict`,
# containing the input and label tensors. Note that :code:`UserDict` is
# :code:`MutableMapping`, so it is compatible with :code:`curvlinops`.

data = next(iter(dataloader))
# print(f"Is the data a UserDict? {isinstance(data, UserDict)}")
# for k, v in data.items():
#     print(k, v.shape)


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
        config = GPT2Config.from_pretrained("gpt2")
        config.pad_token_id = tokenizer.pad_token_id
        config.num_labels = 2
        self.hf_model = GPT2ForSequenceClassification.from_pretrained(
            "gpt2", config=config
        )

        # For simplicity, only enable grad for the last layer
        for p in self.hf_model.parameters():
            p.requires_grad = False

        for p in self.hf_model.score.parameters():
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
            logits: An `(batch_size, n_classes)`-sized tensor of logits.
        """
        device = next(self.parameters()).device
        input_ids = data["input_ids"].to(device)
        attention = data["attention_mask"].to(device)
        output_dict = self.hf_model(input_ids, attention_mask=attention)
        return output_dict.logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyGPT2(tokenizer).to(device=device, dtype=bfloat16)

with no_grad():
    logits = model(data)
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

hessian = HessianLinearOperator(
	model,
	CrossEntropyLoss(),
	params,
	[(data, data["labels"])],
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