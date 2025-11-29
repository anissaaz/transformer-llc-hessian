import os
import re
import copy
import numpy as np
import pandas as pd
import torch

from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from huggingface_hub import list_repo_refs

from datasets import load_dataset
from dataclasses import dataclass, asdict

from collections.abc import MutableMapping
from typing import Union, Literal
from tqdm import tqdm

# --------- Config ---------
MODEL_NAME = "EleutherAI/pythia-14m"
EXPERIMENT_DIR = "llc-batch0-7"
os.makedirs(EXPERIMENT_DIR, exist_ok=True)
MAX_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SGLD Hyperparameters
BATCH_SIZE = 64
TOTAL_DATA_NEEDED = 400 * BATCH_SIZE
SGLD_TEMP = np.log(BATCH_SIZE)

CONFIG = {
    "lr": 1e-5,                     # epsilon (step size)
    "elasticity": 100.0,            # gamma (localization strength)
    "temperature": SGLD_TEMP,
    "num_samples": BATCH_SIZE,
    "num_steps": 400,
    "burnin": 100                   # steps to discard
            
}

@dataclass
class LLCResult:
    revision: str
    step: int
    llc: float
    loss: float

def get_step_tags(model_id: str):
    refs = list_repo_refs(model_id)
    step_refs = []
    for ref in list(refs.tags) + list(refs.branches):
        m = re.fullmatch(r"step(\d+)", ref.name)
        if m:
            step_refs.append((ref.name, int(m.group(1))))
    step_refs.sort(key=lambda x: x[1])
    return step_refs


class MyTransformer(Module):
    """
    Huggingface LLM wrapper.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, model, revision) -> None:
        super().__init__()
        self.hf_model = AutoModelForCausalLM.from_pretrained(model, revision=revision)
        
        # ensure gradients are enabled
        self.hf_model.train()

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
        
        # flatten: (batch * seq, vocab)
        return logits.reshape(logits.shape[0] * logits.shape[1], -1)



def run_analysis(model_name, part=None, output_suffix="full_model"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Preparing Data...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    dataset = load_dataset("EleutherAI/the_pile_deduplicated", split ="train[:1%]")
    subset = dataset.shuffle(seed=0).select(range(TOTAL_DATA_NEEDED))
    texts = [tokenizer.bos_token + ex["text"] + tokenizer.eos_token for ex in subset]
    
    print(f"Tokenizing {len(texts)} samples...")
    batch = tokenizer(
        texts, 
        return_tensors="pt",
        padding = True,
        truncation = True,
        max_length=MAX_LEN,
    )
    
    labels = batch["input_ids"][:,1:].clone()
    input_ids = batch["input_ids"][:,:-1].clone()
    attention_mask = batch["attention_mask"][:,:-1].clone()        # align with input_ids
    
    labels[batch["attention_mask"] == 0] = -100       # labels adjusted for ignore_index=-100 for CE loss
    
    data = TensorDataset(input_ids, attention_mask, labels)
    # DataLoader for minimatches of BATCH_SIZE
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    
    # iterate checkpoints
    step_tags = get_step_tags(model_name)
    step_tags = [
        (rev, step)
        for rev, step in step_tags
        if step in {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 140000}
    ]
    
    loss_fn = CrossEntropyLoss()
    results: list[LLCResult] = []
    
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
        
        
        loss = 
        llc =
        
        
        
        print(f"llc = {llc:.3f}")
        
        results.append(LLC(
            revision=rev,
            step=step,
            llc=llc,
        ))
    
    csv_name = os.path.join(EXPERIMENT_DIR, f"llc_{output_suffix}_0-7.csv")
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(csv_name, index=False, float_format="%.6f")
    print(f"\nSaved results to {csv_name}")
    
if __name__ == "__main__":
    run_analysis("EleutherAI/pythia-14m")