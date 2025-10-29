import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def f(w):
    x, y = w
    return x**3 - 3*x*y**2
f.display_name = r"$f(x,y) = x^3 - 3xy^2$"

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

def hessian_grid_2d(f, xmin=-1, xmax=1, ymin=-1, ymax=1, nx=9, ny=9):
    """
    Evaluate Hessian-based metrics over a 2D grid of points (x,y).
    Returns xs, ys, and metric maps (eigen_max, trace, stable_rank).
    """
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)

    trace_map = np.zeros((ny, nx))
    eigmax_map = np.zeros((ny, nx))
    stable_rank_map = np.zeros((ny, nx))

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            H, eigvals, rank, trace = compute_hessian_and_eigs(f, [x, y])
            stable_rank = compute_stable_rank(torch.from_numpy(H))

            eigmax = np.max(eigvals) if eigvals.size > 0 else np.nan
            trace_val = float(np.real(trace))

            eigmax_map[iy, ix] = eigmax
            trace_map[iy, ix] = trace_val
            stable_rank_map[iy, ix] = stable_rank

    return xs, ys, eigmax_map, trace_map, stable_rank_map

def plot_hessian_heatmap(xs, ys, data_map, title, cmap="viridis", use_percentiles=False):
    extent = [xs.min(), xs.max(), ys.min(), ys.max()]
    
    data = data = np.asarray(data_map, dtype=float)
    
    # Keep zeros/undefined as NaN so they render white
    mask = ~np.isfinite(data)
    data_for_plot = data.copy()
    data_for_plot[mask] = np.nan
    
    # Choose color scale from positive, finite values only
    vals = data[np.isfinite(data) & (data != 0)]
    if vals.size == 0:
        # nothing to scale from; just show the masked image
        vmin, vmax = None, None
    else:
        if use_percentiles:
            vmin = np.percentile(vals, 1)   # or 0.5
            vmax = np.percentile(vals, 99)  # or 99.5
            if vmin == vmax:
                vmin, vmax = vals.min(), vals.max()
        else:
            vmin, vmax = vals.min(), vals.max()
            if vmin == vmax:
                # degenerate case: expand a hair to avoid a flat colorbar
                eps = 1e-12
                vmin, vmax = vmin, vmin + eps
    
    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        data_for_plot,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax) if (vmin is not None) else None,
        interpolation="nearest",
    )
    
    plt.colorbar(im, label=title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    xs, ys, eigmax_map, trace_map, stable_rank_map = hessian_grid_2d(f, xmin=-1, xmax=1, ymin=-1, ymax=1, nx=15, ny=15)

    plot_hessian_heatmap(xs, ys, eigmax_map, title=f"Largest Eigenvalue of Hessian for {getattr(f, 'display_name', f.__name__)}")
    plot_hessian_heatmap(xs, ys, trace_map, title=f"Hessian Trace for {getattr(f, 'display_name', f.__name__)}")
    plot_hessian_heatmap(xs, ys, stable_rank_map, title=f"Hessian Stable Rank for {getattr(f, 'display_name', f.__name__)}", use_percentiles=True)