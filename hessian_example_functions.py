import torch
import numpy as np

def f(w):
    x, y = w
    return (x**2) * (y**4)

def compute_hessian_and_eigs(f, w0):
    w = torch.tensor(w0, dtype=torch.float64, requires_grad=True)
    y = f(w)
    grad = torch.autograd.grad(y, w, create_graph=True)[0]              # gradient vector
    H_rows = []
    for g_i in grad:                                                    # gradient of each component in vector gradient computed
        H_row = torch.autograd.grad(g_i, w, retain_graph=True)[0]
        H_rows.append(H_row)
    H = torch.stack(H_rows)
    eigvals, eigvecs = torch.linalg.eig(H)
    eigvals = eigvals.real  # discard imaginary values
    rank = (eigvals.abs() > 1e-10).sum().item()
    trace = torch.trace(H)
    return H.detach().cpu().numpy(), eigvals.detach().cpu().numpy(), rank, trace.detach().cpu().numpy()

def compute_stable_rank(H):
    frobenius_norm = torch.linalg.matrix_norm(H, ord='fro')
    spectral_norm = torch.linalg.matrix_norm(H, ord=2)
    stable_rank = (frobenius_norm**2) / (spectral_norm**2 + 1e-12)
    return stable_rank.item()

H, eigvals, rank, trace = compute_hessian_and_eigs(f, [1.0, 2.0])
stable_rank = compute_stable_rank(torch.from_numpy(H))

print("Hessian:\n", H)
print("Eigenvalues:", eigvals)
print("Rank:", rank)
print(f"Stable rank: {stable_rank:.4f}")
print("Trace: ", trace)