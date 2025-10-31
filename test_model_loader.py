import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "EleutherAI/pythia-160m"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
model.eval()

# Get logits + hidden states
text = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer(text, return_tensors="pt").to(device)

# Ask the model to return hidden states
outputs = model(**inputs, output_hidden_states=True)

# Take the last hidden state for the last token
# shape: (batch=1, d_model)
h_last = outputs.hidden_states[-1][:, -1, :].detach().requires_grad_(True)

# Output projection (this is the 'lm_head' for GPT-NeoX)
lm_head = model.get_output_embeddings()   # == model.embed_out

# Logits for the last position from the last hidden vector
logits_last = lm_head(h_last)  # shape: (1, vocab_size)

# Target is the next token after the last input token
target = inputs["input_ids"][:, -1]  # teacher-forcing style (simple demo)
loss = F.cross_entropy(logits_last, target)

# ---- Build Hessian wrt h_last (2048 x 2048) ----
grad = torch.autograd.grad(loss, h_last, create_graph=True)[0].squeeze(0)  # (d_model,)

H_rows = []
for g_i in grad:  # loop over d_model
    H_row = torch.autograd.grad(g_i, h_last, retain_graph=True)[0].squeeze(0)  # (d_model,)
    H_rows.append(H_row)
H = torch.stack(H_rows)  # (d_model, d_model)

# Symmetrize (numerical nicety)
H = 0.5 * (H + H.T)

# Eigen stuff
eigvals = torch.linalg.eigvalsh(H).detach()
max_eig = eigvals.max()
fro_sq = torch.sum(eigvals**2)          # == ||H||_F^2 for symmetric H
stable_rank = (fro_sq / (max_eig**2)).item()
trace = eigvals.sum().item()

print("Max eigenvalue:", max_eig.item())
print("Stable rank:", stable_rank)
print("Trace:", trace)