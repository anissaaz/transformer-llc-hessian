import numpy as np

def alg1_llc_estimate(
    f, grad_f, w_star,
    n=200,            # dataset size (only used in beta* = 1/log n and n factor)
    gamma=10.0,       # localization scale (keeps chain near w*)
    eps=1e-3,         # SGLD step size "epsilon" in the pseudo-code
    iters=40000,      # total SGLD iterations
    burn=10000,       # discard first steps
    seed=0
):
    """
    Implements Algorithm 1 from the paper for toy losses by setting logL = -f.
    Returns (lambda_hat, wbic, logL_mean).
    """
    rng = np.random.default_rng(seed)
    w = np.array(w_star, dtype=np.float64)
    d = w.size

    beta_star = 1.0 / np.log(n)                    # line 1: β* = 1/log n

    logL_trace = []
    f_min_seen = np.inf
    
    fw_star = float(f(w_star))
    
    # diagnostic, if ≈ 0 -> w_star near local minimum
    grad_fw_star = grad_f(np.array(w_star, dtype=np.float64))
    grad_fw_star_norm = float(np.linalg.norm(grad_fw_star))             
    # print(f"Pre-run diagnostic: f(w*) = {fw_star:.6g}, ||grad f(w*)|| = {grad_fw_star_norm:.6g}")


    for t in range(iters):
        # track minimum f seen so far
        fw = float(f(w))
        if fw < f_min_seen:
            f_min_seen = fw
        
        # Langevin steps:
        # line 5: sample minibatch B (here: just evaluate at w)
        logL = -f(w)                               # line 6: append logL(B,w), no dataset here
       
        # line 7: eta ~ N(0, epsilon)
        eta = rng.normal(size=d) * np.sqrt(eps)

        # line 8: Δw = (eps/2)[gamma(w* - w) + n beta* ∇_w logL(B,w)] + eta
        grad_logL = -grad_f(w)                     # ∇ logL = -∇ f
        delta = 0.5 * eps * (gamma*(w_star - w) + n*beta_star*grad_logL) + eta

        # line 9: w <- w + Δw
        w = w + delta
        
        if t >= burn:                              # only record after burn-in
            logL_trace.append(logL)              
    
    # line 11: WBIC = - n * mean(arrayLogL)
    logL_mean = np.mean(logL_trace)
    wbic = -n * logL_mean  # since logL is negative, -n*mean(logL) = n * E[f(w)]

    # line 12: n L_n(w*) = -n * logL(D_n, w*) = n * f(w*); for toy f, f(w*) = 0
    n_Ln_wstar = n * f(w_star)

    # line 13: lambda_hat = (WBIC - n L_n(w*)) / log n
    lambda_hat = (wbic - n_Ln_wstar) / np.log(n)
    
    # print(f"Min f seen during chain: {f_min_seen:.4g}")

    return float(lambda_hat), float(wbic), float(logL_mean)


# ---------- Examples ----------
# 1D:

def setup_quadratic():
    f = lambda w: w[0]**2
    grad_f = lambda w: np.array([2.0*w[0]], dtype=np.float64)
    w_star = [0.0]
    est = dict(n=200, gamma=2.0, eps=5e-4, iters=80_000, burn=15_000, seed=1)
    return f, grad_f, w_star, est

def setup_quartic():
    f = lambda w: w[0]**4
    grad_f = lambda w: np.array([4.0*w[0]**3], dtype=np.float64)
    w_star = [0.0]
    est = dict(n=200, gamma=2.0, eps=5e-4, iters=80_000, burn=15_000, seed=0)
    return f, grad_f, w_star, est

# 2D:

def setup_x2_plus_y2():
    f = lambda w: w[0]**2 + w[1]**2
    grad_f = lambda w: np.array([2.0*w[0], 2.0*w[1]], dtype=np.float64)
    w_star = [0.0, 0.0]
    est = dict(n=300, gamma=2.0, eps=2e-4, iters=150_000, burn=100_000, seed=3)
    return f, grad_f, w_star, est

def setup_x4_plus_y4():
    f = lambda w: w[0]**4 + w[1]**4
    grad_f = lambda w: np.array([4.0*w[0]**3, 4*w[1]**3], dtype=np.float64)
    w_star = [0.0, 0.0]
    est = dict(n=300, gamma=2.0, eps=1e-4, iters=250_000, burn=120_000, seed=4)
    return f, grad_f, w_star, est

def setup_x2y4():
    f = lambda w: (w[0]**2)*(w[1]**4)
    grad_f = lambda w: np.array([2*w[0]*(w[1]**4), 4*(w[0]**2)*(w[1]**3)], dtype=np.float64)
    w_star = [0.0, 0.0]
    est = dict(n=300, gamma=1.0, eps=1e-4, iters=250_000, burn=105_000, seed=2)
    return f, grad_f, w_star, est

def setup_x3_minus_3xy2():
    f = lambda w: (w[0]**3 - 3*w[0]*(w[1]**2))
    grad_f = lambda w: np.array([3*(w[0]**2) - 3*(w[1]**2), -6*w[0]*w[1]], dtype=np.float64)
    w_star = [0.0, 0.0]
    est = dict(n=300, gamma=2.0, eps=2e-5, iters=200_000, burn=100_000, seed=6)
    return f, grad_f, w_star, est
# ---------- N-D (3D and higher) helpers ----------

def make_separable_even_powers(powers):
    """
    Build (f, grad_f, w_star) for f(w) = sum_i |w_i|^{p_i} with even integers p_i (e.g., [2,2,2] or [4,4,2,2]).
    The gradient is elementwise: ∂f/∂w_i = p_i * w_i^{p_i-1}.
    """
    powers = list(powers)
    d = len(powers)

    def f(w):
        w = np.asarray(w, dtype=np.float64)
        return sum((w[i] ** powers[i]) for i in range(d))

    def grad_f(w):
        w = np.asarray(w, dtype=np.float64)
        return np.array([powers[i] * (w[i] ** (powers[i] - 1)) for i in range(d)], dtype=np.float64)

    w_star = [0.0] * d
    return f, grad_f, w_star

def setup_3d_quadratic():
    """
    3D example: f(x,y,z) = x^2 + y^2 + z^2 (theoretical LLC at the minimum = 3 * (1/2) = 1.5)
    """
    f, grad_f, w_star = make_separable_even_powers([2, 2, 2])
    est = dict(n=300, gamma=2.0, eps=2e-4, iters=180_000, burn=90_000, seed=5)
    return f, grad_f, w_star, est

def setup_nd_separable(powers, n=300, gamma=2.0, eps=2e-4, iters=180_000, burn=90_000, seed=0):
    """
    Generic N-D separable sum of even powers: powers like [2,2,4,4,...].
    """
    f, grad_f, w_star = make_separable_even_powers(powers)
    est = dict(n=n, gamma=gamma, eps=eps, iters=iters, burn=burn, seed=seed)
    return f, grad_f, w_star, est


def run_quadratic():
    f, grad_f, w_star, est = setup_quadratic()
    lam, wbic, _ = alg1_llc_estimate(f, grad_f, w_star=w_star, **est)
    print(f"[w^2]  lambda_hat ≈ {lam:.3f}   (theory 0.50)")

def run_quartic():
    f, grad_f, w_star, est = setup_quartic()
    lam, wbic, _ = alg1_llc_estimate(f, grad_f, w_star=w_star, **est)
    print(f"[w^4]  lambda_hat ≈ {lam:.3f}   (theory 0.25)")

def run_x2_plus_y2():
    f, grad_f, w_star, est = setup_x2_plus_y2()
    lam, wbic, _ = alg1_llc_estimate(f, grad_f, w_star=w_star, **est)
    print(f"[x^2 + y^2]  lambda_hat ≈ {lam:.3f}   (theory 1.00)")
    
def run_x4_plus_y4():
    f, grad_f, w_star, est = setup_x4_plus_y4()
    lam, wbic, _ = alg1_llc_estimate(f, grad_f, w_star=w_star, **est)
    print(f"[x^4 + y^4]  lambda_hat ≈ {lam:.3f}   (theory 0.50)")

def run_x2y4():
    f, grad_f, w_star, est = setup_x2y4()
    lam, wbic, _ = alg1_llc_estimate(f, grad_f, w_star=w_star, **est)
    print(f"[x^2 y^4]  lambda_hat ≈ {lam:.3f}   (theory 0.25)")

def run_x3_minus_3xy2():
    f, grad_f, w_star, est = setup_x3_minus_3xy2()
    lam, wbic, _ = alg1_llc_estimate(f, grad_f, w_star=w_star, **est)
    print(r"$f(x,y) = x^3 - 3xy^2$" + f"lambda_hat ≈ {lam:.3f}")

# --------- N-D demo runners ----------
def run_3d_quadratic():
    f, grad_f, w_star, est = setup_3d_quadratic()
    lam, wbic, _ = alg1_llc_estimate(f, grad_f, w_star=w_star, **est)
    print(f"[x^2 + y^2 + z^2]  lambda_hat ≈ {lam:.3f}   (theory 1.50)")

def run_nd_separable_example(powers, **est_overrides):
    """
    Run a single-point LLC estimate for f(w)=sum_i |w_i|^{p_i} at w*=0 in any dimension.
    'powers' is a list like [2,2,4,4,2].
    """
    f, grad_f, w_star, est = setup_nd_separable(powers, **est_overrides)
    lam, wbic, _ = alg1_llc_estimate(f, grad_f, w_star=w_star, **est)
    theory = sum(1.0 / p for p in powers)  # separable sum: theoretical lambda = sum 1/degree_i
    print(f"[sum |w|^p, p={powers}]  lambda_hat ≈ {lam:.3f}   (theory {theory:.2f})")



if __name__ == "__main__":
    run_quadratic()
    run_quartic()
    run_x2_plus_y2()
    run_x4_plus_y4()
    run_x2y4()
    run_x3_minus_3xy2()
    run_3d_quadratic()
    #run_nd_separable_example([2,2,4, 3])

# ---------- 2‑D LLC heat map ----------

def llc_grid_2d(
    f, grad_f,
    xmin, xmax, ymin, ymax,                           # rectangle for LLC measuring
    nx=5, ny=5,                                       # grid resolution (number of points along x and y)
    **est_kwargs                                      # forwarded to alg1_llc_estimate (n, gamma, eps, iters, burn, seed, ...)
):
    """
    Sweep LLC estimates over a 2-D grid of initialization points w*=(x,y).
    Returns (xs, ys, lam_map) with lam_map shape = (ny, nx).

    """
    
    # Defaults for estimator hyperparameters; can be overridden via **est_kwargs
    defaults = dict(n=500, gamma=2.0, eps=2e-4, iters=120_000, burn=35_000, seed=0)
    defaults.update(est_kwargs)

    # builds evenly spaced sample points on each axis
    rng = np.random.default_rng(defaults["seed"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    
    lam_mean_map = np.zeros((ny, nx), dtype=float)
    lam_std_map  = np.zeros((ny, nx), dtype=float)


    # for reproducibility: vary seed per cell to decorrelate chains
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            
            K = 10  # number of SGLD chains per grid cell
            lam_values = []
            for k in range(K):
                defaults["seed"] = int(rng.integers(0, 2**31 - 1))
                lam, _, _ = alg1_llc_estimate(
                    f, grad_f, w_star=[x, y],
                    **defaults
                )
                lam_values.append(lam)
            lam_values = np.asarray(lam_values, dtype=float)
            lam_mean_map[iy, ix] = lam_values.mean()  # average over K chains per point
            lam_std_map[iy, ix]  = lam_values.std(ddof=1)
            
            
            # cell_seed = int(rng.integers(0, 2**31-1))
            # defaults["seed"] = cell_seed
            # lam, _, _ = alg1_llc_estimate(
            #     f, grad_f, w_star=[x, y],
            #     **defaults
            # )
            # lam_map[iy, ix] = lam                   # one scalar llc per point
    return xs, ys, lam_mean_map, lam_std_map


def plot_llc_heatmap(xs, ys, lam_map, title="LLC heatmap", cmap="viridis"):
    """heatmap utility for lam_map with axes labeled by xs, ys."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plt.figure()
    extent = [xs.min(), xs.max(), ys.min(), ys.max()]
    plt.imshow(lam_map, origin="lower", extent=extent, aspect="auto",
               interpolation="nearest", cmap=cmap, norm=Normalize(vmin=np.nanmin(lam_map), vmax=np.nanmax(lam_map)))
    plt.colorbar(label=r"$\hat{\lambda}(w^*)$")
    plt.xlabel("x (w* component 1)")
    plt.ylabel("y (w* component 2)")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    
def plot_llc_errorbars(xs, ys, lam_mean_map, lam_std_map, title="LLC with error bars"):
    """
    Plot mean ± (std or SE) as error bars for each grid cell (flattened).
    - use_se=False -> error bars = standard deviation (spread of chain estimates)
    - use_se=True  -> error bars = standard error = std/sqrt(K) (uncertainty of the mean)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    ny, nx = lam_mean_map.shape
    Xg, Yg = np.meshgrid(xs, ys)
    means = lam_mean_map.ravel()
    stds  = lam_std_map.ravel()

    errs = stds

    # Nice labels (x,y) for the x-axis; you can also keep them numeric if there are many points.
    labels = [f"({Xg.ravel()[i]:.3f},{Yg.ravel()[i]:.3f})" for i in range(means.size)]

    plt.figure(figsize=(max(8, means.size*0.25), 5))
    xind = np.arange(means.size)
    plt.errorbar(xind, means, yerr=errs, fmt='o', capsize=3, linewidth=1)
    plt.xticks(xind, labels, rotation=90)
    plt.ylabel(r"$\hat{\lambda}(w^*)$")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    
# --------- Diagonal LLC sweep plotting ----------
def plot_diagonal_from_grid(xs, ys, lam_mean_map, lam_std_map, title="Diagonal LLC sweep from grid"):
    """
    Plot mean ± std as error bars along the diagonal of the grid (where x==y indices).
    - xs, ys: 1D arrays of grid coordinates (same as returned by llc_grid_2d)
    - lam_mean_map, lam_std_map: 2D arrays of mean/std per grid point
    - title: plot title
    """
    import numpy as np
    import matplotlib.pyplot as plt

    ny, nx = lam_mean_map.shape
    diag_len = min(nx, ny)
    # Extract diagonal values: index pairs (i, i)
    diag_xs = np.array([xs[i] for i in range(diag_len)])
    diag_means = np.array([lam_mean_map[i, i] for i in range(diag_len)])
    diag_stds  = np.array([lam_std_map[i, i] for i in range(diag_len)])

    diag_errs = diag_stds

    plt.figure(figsize=(max(6, diag_len * 0.6), 4))
    plt.errorbar(diag_xs, diag_means, yerr=diag_errs, fmt='o-', capsize=4, linewidth=2)
    plt.xlabel("r (diagonal coordinate)")
    plt.ylabel(r"$\hat{\lambda}(w^*)$")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def run_grid_x2_plus_y2_demo():
    f, grad_f, _, est = setup_x2_plus_y2()
    xs, ys, lam_mean_map, lam_std_map = llc_grid_2d(
        f, grad_f,
        xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5,
        nx=9, ny=9,
        **est
    )
    plot_llc_heatmap(xs, ys, lam_mean_map, title="LLC over (x,y) for f(x,y)=x^2+y^2")
    plot_llc_errorbars(xs, ys, lam_mean_map, lam_std_map, title="Per-point LLC with error bars (std)")
    plot_diagonal_from_grid(xs, ys, lam_mean_map, lam_std_map, title="Diagonal LLC sweep from grid")
    
    
def run_grid_x4_plus_y4_demo():
    f, grad_f, _, est = setup_x4_plus_y4()
    xs, ys, lam_mean_map, lam_std_map = llc_grid_2d(
        f, grad_f,
        xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5,
        nx=9, ny=9,
        **est
    )
    plot_llc_heatmap(xs, ys, lam_mean_map, title="LLC mean over (x,y) for f(x,y)=x^4+y^4")
    plot_llc_errorbars(xs, ys, lam_mean_map, lam_std_map, title="Per-point LLC with error bars (std)")

def run_grid_x2y4_demo():
    f, grad_f, _, est = setup_x2y4()
    xs, ys, lam_mean_map, lam_std_map = llc_grid_2d(
        f, grad_f,
        xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5,
        nx=9, ny=9,
        **est
    )
    plot_llc_heatmap(xs, ys, lam_mean_map, title="LLC over (x,y) for f(x,y)=x²y⁴")
    plot_llc_errorbars(xs, ys, lam_mean_map, lam_std_map, title="Per-point LLC with error bars (std)")
    plot_diagonal_from_grid(xs, ys, lam_mean_map, lam_std_map, title="Diagonal LLC sweep from grid")

def run_grid_x3_minus_3xy2_demo():
    f, grad_f, _, est = setup_x3_minus_3xy2()
    xs, ys, lam_mean_map, lam_std_map = llc_grid_2d(
        f, grad_f,
        xmin=-0.6, xmax=0.6, ymin=-0.6, ymax=0.6,
        nx=9, ny=9,
        **est
    )
    plot_llc_heatmap(xs, ys, lam_mean_map,
                     title=r"LLC over (x,y) for $f(x,y)=x^3-3xy^2$")
    plot_llc_errorbars(xs, ys, lam_mean_map, lam_std_map,
                       title=r"Per-point LLC with error bars (std) for $x^3-3xy^2$")
    plot_diagonal_from_grid(xs, ys, lam_mean_map, lam_std_map,
                            title=r"Diagonal LLC sweep for $x^3-3xy^2$")

# ---------- 3D slice visualizations (reuse 2-D grids) ----------

def restrict_xy_at_z(f3, grad3, z0):
    """
    Freeze z=z0 and return 2D (f, grad_f) in (x,y).
    """
    def f2(u):
        x, y = float(u[0]), float(u[1])
        return f3(np.array([x, y, z0], dtype=np.float64))

    def grad2(u):
        x, y = float(u[0]), float(u[1])
        g3 = grad3(np.array([x, y, z0], dtype=np.float64))
        return np.array([g3[0], g3[1]], dtype=np.float64)

    return f2, grad2

def restrict_xz_at_y(f3, grad3, y0):
    """
    Freeze y=y0 and return 2D (f, grad_f) in (x,z).
    """
    def f2(u):
        x, z = float(u[0]), float(u[1])
        return f3(np.array([x, y0, z], dtype=np.float64))

    def grad2(u):
        x, z = float(u[0]), float(u[1])
        g3 = grad3(np.array([x, y0, z], dtype=np.float64))
        return np.array([g3[0], g3[2]], dtype=np.float64)

    return f2, grad2

def restrict_yz_at_x(f3, grad3, x0):
    """
    Freeze x=x0 and return 2D (f, grad_f) in (y,z).
    """
    def f2(u):
        y, z = float(u[0]), float(u[1])
        return f3(np.array([x0, y, z], dtype=np.float64))

    def grad2(u):
        y, z = float(u[0]), float(u[1])
        g3 = grad3(np.array([x0, y, z], dtype=np.float64))
        return np.array([g3[1], g3[2]], dtype=np.float64)

    return f2, grad2

def run_3d_quadratic_slices_demo():
    """
    Example 3D visualization: f=x^2+y^2+z^2.
    We show XY-slices at several z values.
    """
    f3, grad3, _, est = setup_3d_quadratic()

    # Choose slice levels for the frozen coord and the 2D window/grid.
    z_levels = [-0.3, 0.0, 0.3]
    xmin, xmax, ymin, ymax = -0.6, 0.6, -0.6, 0.6
    nx, ny = 9, 9

    for z0 in z_levels:
        f2, g2 = restrict_xy_at_z(f3, grad3, z0)
        xs, ys, lam_mean_map, lam_std_map = llc_grid_2d(
            f2, g2,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            nx=nx, ny=ny,
            **est
        )
        plot_llc_heatmap(xs, ys, lam_mean_map, title=f"LLC slice on XY at z={z0:+.2f}")

run_grid_x4_plus_y4_demo()
#run_grid_x2_plus_y2_demo()
#run_grid_x2y4_demo()
#run_3d_quadratic_slices_demo()     # XY-slice heatmaps for 3D quadratic
#run_grid_x3_minus_3xy2_demo()