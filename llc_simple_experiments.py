import numpy as np

def alg1_llc_estimate(
    f, grad_f, w_star,
    n=200,            # dataset size (only used in beta* = 1/log n and n factor)
    gamma=20.0,       # localization scale (keeps chain near w*)
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

    for t in range(iters):
        # line 5: sample minibatch B (here: just evaluate at w)
        logL = -f(w)                               # line 6: append logL(B,w), no dataset here
        if t >= burn:                              # only record after burn-in
            logL_trace.append(logL)

        # line 7: eta ~ N(0, epsilon)
        eta = rng.normal(size=d) * np.sqrt(eps)

        # line 8: Δw = (eps/2)[gamma(w* - w) + n beta* ∇_w logL(B,w)] + eta
        grad_logL = -grad_f(w)                     # ∇ logL = -∇ f
        delta = 0.5 * eps * (gamma*(w_star - w) + n*beta_star*grad_logL) + eta

        # line 9: w <- w + Δw
        w = w + delta

    # line 11: WBIC = - n * mean(arrayLogL)
    logL_mean = np.mean(logL_trace)
    wbic = -n * logL_mean  # since logL is negative, -n*mean(logL) = n * E[f(w)]

    # line 12: n L_n(w*) = -n * logL(D_n, w*) = n * f(w*); for toy f, f(w*) = 0
    n_Ln_wstar = n * f(w_star)

    # line 13: lambda_hat = (WBIC - n L_n(w*)) / log n
    lambda_hat = (wbic - n_Ln_wstar) / np.log(n)

    return float(lambda_hat), float(wbic), float(logL_mean)

# ---------- Examples ----------

def run_quartic():
    # 1D: f(w)=w^4, ∇f=4 w^3, true LLC = 1/4
    f = lambda w: w[0]**4
    grad_f = lambda w: np.array([4.0 * w[0]**3], dtype=np.float64)
    lam, wbic, logL_mean = alg1_llc_estimate(
        f, grad_f, w_star=[0.0],
        n=200, gamma=2.0, eps=5e-4, iters=60000, burn=15000, seed=0
    )
    print(f"[w^4]  lambda_hat ≈ {lam:.3f}   (theory 0.25)")

def run_quadratic():
    # 1D: f(w)=w^2, true LLC = 1/2
    f = lambda w: w[0]**2
    grad_f = lambda w: np.array([2.0 * w[0]], dtype=np.float64)
    lam, wbic, logL_mean = alg1_llc_estimate(
        f, grad_f, w_star=[0.0],
        n=200, gamma=2.0, eps=5e-4, iters=60000, burn=15000, seed=1
    )
    print(f"[w^2]  lambda_hat ≈ {lam:.3f}   (theory 0.50)")

def run_x2y4():
    # 2D: f(x,y)=x^2 y^4, true LLC = 1/4
    f = lambda w: (w[0]**2)*(w[1]**4)
    grad_f = lambda w: np.array([ 2*w[0]*(w[1]**4),  4*(w[0]**2)*(w[1]**3) ], dtype=np.float64)
    lam, wbic, logL_mean = alg1_llc_estimate(
        f, grad_f, w_star=[0.0, 0.0],
        n=300, gamma=2.0, eps=3e-4, iters=80000, burn=20000, seed=2
    )
    print(f"[x^2 y^4]  lambda_hat ≈ {lam:.3f}   (theory 0.25)")

if __name__ == "__main__":
    run_quartic()
    run_quadratic()
    run_x2y4()