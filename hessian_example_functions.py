import torch

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
    return H.detach().numpy(), eigvals.detach().numpy(), rank

H, eigvals, rank = compute_hessian_and_eigs(f, [1.0, 2.0])
print("Hessian:\n", H)
print("Eigenvalues:", eigvals)
print("Rank:", rank)