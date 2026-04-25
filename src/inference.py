"""
src/inference.py
================
Hawkes process parameter inference via the Expectation-Maximization
algorithm.  All estimators follow the branching-process representation
of Hawkes (1971): each observed event is either a background arrival
(Poisson rate mu) or an offspring triggered by a prior event.

Public API
----------
branching_em(events, T, ...)
    Single-component (K=1) Branching EM.
    Returns (mu, alpha, beta, ll_history, P).

branching_em_with_diagnostics(events, T, ...)
    Same algorithm, additionally returns the full parameter trajectory
    and step-size sequence for convergence analysis.
    Returns (mu, alpha, beta, ll_history, param_history, param_diffs, P).

bimodal_branching_em(events, T, ...)
    Two-component (K=2) Branching EM.
    Returns (mus, alphas, betas, R_bg, R_trig).

compute_bic(ll, n_params, n_samples)
    Bayesian Information Criterion helper.
"""

import numpy as np
from scipy.optimize import minimize


# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────

def _enforce_stability(alpha, beta, eps=1e-6):
    """Clip parameters to the interior of the stability region alpha < beta."""
    beta  = max(beta,  eps)
    alpha = min(alpha, 0.99 * beta)
    alpha = max(alpha, eps)
    return alpha, beta


def _compensator(events, mu, alpha, beta, T):
    """
    Closed-form integral of lambda(t) over [0, T]:
        integral = mu*T + (alpha/beta) * sum_i [1 - exp(-beta*(T - t_i))]
    """
    return mu * T + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - events)))


def _build_branching_matrix(events, mu, alpha, beta):
    """
    Construct the (n x n) branching probability matrix P.

    P[i, i] = mu / lambda(t_i)
    P[i, j] = alpha * exp(-beta*(t_i - t_j)) / lambda(t_i)  for j < i

    Each row sums to 1.  Returns (P, lam_t) where lam_t[i] = lambda(t_i).
    """
    n        = len(events)
    dt       = events[:, None] - events[None, :]
    lower    = dt > 0

    L        = np.zeros((n, n))
    L[lower] = alpha * np.exp(-beta * dt[lower])
    np.fill_diagonal(L, mu)

    lam_t = np.maximum(L.sum(axis=1), 1e-300)
    P     = L / lam_t[:, None]
    return P, lam_t


def compute_bic(ll, n_params, n_samples):
    """BIC = -2 * log_likelihood + n_params * log(n_samples)."""
    return -2.0 * ll + n_params * np.log(n_samples)


# ─────────────────────────────────────────────────────────────────────
# K = 1  Branching EM
# ─────────────────────────────────────────────────────────────────────

def branching_em(events, T, max_iter=100, tol=1e-6):
    """
    Single-component Branching EM for a Hawkes process on [0, T].

    Parameters
    ----------
    events   : 1-D sorted array of event times in (0, T).
    T        : observation horizon.
    max_iter : maximum EM iterations.
    tol      : convergence threshold on log-likelihood increment.

    Returns
    -------
    mu, alpha, beta : final parameter estimates (floats).
    ll_history      : list of log-likelihood values, one per iteration.
    P               : final branching matrix, shape (n, n).
                      P[i, i]  = background probability for event i.
                      P[i, j]  = probability that event j triggered event i.
    """
    n           = len(events)
    mu          = n / T
    alpha, beta = 0.2, 1.0
    ll_history  = []

    dt      = events[:, None] - events[None, :]
    lower   = dt > 0
    dt_vals = dt[lower]

    for it in range(max_iter):

        # ── E-step ──
        P, lam_t = _build_branching_matrix(events, mu, alpha, beta)

        # ── Log-likelihood ──
        ll = np.sum(np.log(lam_t)) - _compensator(events, mu, alpha, beta, T)
        ll_history.append(float(ll))

        # ── M-step: mu (analytical) ──
        mu = max(np.diag(P).sum() / T, 1e-8)

        # ── M-step: alpha, beta (numerical) ──
        p_sum = np.tril(P, -1).sum()
        w_dt  = (P[lower] * dt_vals).sum()

        def _neg_obj(x, _p=p_sum, _w=w_dt, _ev=events):
            a, b = x
            if a <= 0 or b <= 0 or a >= b:
                return 1e10
            return -(
                _p * np.log(a)
                - b * _w
                - (a / b) * np.sum(1.0 - np.exp(-b * (T - _ev)))
            )

        res         = minimize(_neg_obj, [alpha, beta],
                               method='Nelder-Mead',
                               options={'maxiter': 400, 'xatol': 1e-6,
                                        'fatol': 1e-6})
        alpha, beta = _enforce_stability(*res.x)

        if it > 5 and (ll_history[-1] - ll_history[-2]) < tol:
            break

    P_final, _ = _build_branching_matrix(events, mu, alpha, beta)
    return mu, alpha, beta, ll_history, P_final


# ─────────────────────────────────────────────────────────────────────
# K = 1  Branching EM with full convergence diagnostics
# ─────────────────────────────────────────────────────────────────────

def branching_em_with_diagnostics(events, T, max_iter=200, tol=1e-8):
    """
    Identical to branching_em but records the full parameter trajectory
    and per-iteration step sizes for convergence analysis.

    The tolerance is intentionally tighter (1e-8) so the convergence
    curve covers enough iterations to fit a linear convergence rate.

    Returns
    -------
    mu, alpha, beta  : final estimates.
    ll_history       : list of log-likelihood values.
    param_history    : ndarray shape (n_iters, 3), columns = [mu, alpha, beta].
    param_diffs      : ndarray shape (n_iters - 1,),
                       ||theta^{n+1} - theta^n||_2 at each iteration.
    P                : final branching matrix, shape (n, n).
    """
    n           = len(events)
    mu          = n / T
    alpha, beta = 0.2, 1.0
    ll_history  = []
    param_history = []

    dt      = events[:, None] - events[None, :]
    lower   = dt > 0
    dt_vals = dt[lower]

    for it in range(max_iter):

        # ── E-step ──
        P, lam_t = _build_branching_matrix(events, mu, alpha, beta)

        # ── Log-likelihood ──
        ll = np.sum(np.log(lam_t)) - _compensator(events, mu, alpha, beta, T)
        ll_history.append(float(ll))
        param_history.append(np.array([mu, alpha, beta]))

        # ── M-step: mu ──
        mu = max(np.diag(P).sum() / T, 1e-8)

        # ── M-step: alpha, beta ──
        p_sum = np.tril(P, -1).sum()
        w_dt  = (P[lower] * dt_vals).sum()

        def _neg_obj(x, _p=p_sum, _w=w_dt, _ev=events):
            a, b = x
            if a <= 0 or b <= 0 or a >= b:
                return 1e10
            return -(
                _p * np.log(a)
                - b * _w
                - (a / b) * np.sum(1.0 - np.exp(-b * (T - _ev)))
            )

        res         = minimize(_neg_obj, [alpha, beta],
                               method='Nelder-Mead',
                               options={'maxiter': 400, 'xatol': 1e-7,
                                        'fatol': 1e-7})
        alpha, beta = _enforce_stability(*res.x)

        if it > 10 and (ll_history[-1] - ll_history[-2]) < tol:
            break

    param_history = np.array(param_history)
    param_diffs   = np.linalg.norm(np.diff(param_history, axis=0), axis=1)

    P_final, _ = _build_branching_matrix(events, mu, alpha, beta)
    return mu, alpha, beta, ll_history, param_history, param_diffs, P_final


# ─────────────────────────────────────────────────────────────────────
# K = 2  Bimodal Branching EM
# ─────────────────────────────────────────────────────────────────────

def bimodal_branching_em(events, T, max_iter=100, tol=1e-6):
    """
    Two-component Bimodal Branching EM.

    The merged event stream is attributed softly to two latent Hawkes
    components.  The soft assignment of event i to component k is:

        total_resp_k[i]  =  lambda_k(t_i) / lambda_total(t_i)

    which decomposes into background and offspring contributions:

        R_bg[k, i]       =  mu_k          / lambda_total(t_i)
        R_trig[k, i, j]  =  alpha_k * exp(-beta_k*(t_i - t_j))
                            / lambda_total(t_i)   for j < i

    so that  R_bg[k, i] + R_trig[k, i, :].sum()  ==  total_resp_k[i].

    Parameters
    ----------
    events   : 1-D sorted array of event times.
    T        : observation horizon.
    max_iter : maximum EM iterations.
    tol      : convergence threshold on log-likelihood increment.

    Returns
    -------
    mus    : ndarray (2,)   — background intensities [mu_0, mu_1].
    alphas : ndarray (2,)   — excitation strengths   [alpha_0, alpha_1].
    betas  : ndarray (2,)   — decay rates            [beta_0, beta_1].
    R_bg   : ndarray (2, n) — background responsibilities.
             R_bg[k, i]  = P(event i is background from component k).
    R_trig : ndarray (2, n, n) — offspring responsibilities.
             R_trig[k, i, j]  = P(event i was triggered by event j,
                                   both attributed to component k),
             non-zero only for i > j.
    """
    n = len(events)

    # Initialise: component 0 = slow background, component 1 = fast
    mus    = np.array([n / T * 0.4,  n / T * 1.2])
    alphas = np.array([0.10, 0.70])
    betas  = np.array([0.80, 3.00])

    dt    = events[:, None] - events[None, :]
    lower = dt > 0

    ll_history = []

    for it in range(max_iter):

        # ── E-step ──
        # Build per-component intensity matrices (n x n)
        L = np.zeros((2, n, n))
        for k in range(2):
            L[k][lower] = alphas[k] * np.exp(-betas[k] * dt[lower])
            np.fill_diagonal(L[k], mus[k])

        lam_total = np.maximum(L[0].sum(axis=1) + L[1].sum(axis=1), 1e-300)

        # Responsibility matrices
        R_bg   = np.zeros((2, n))
        R_trig = np.zeros((2, n, n))
        for k in range(2):
            R_bg[k] = np.diag(L[k]) / lam_total
            R_trig[k] = L[k] / lam_total[:, None]
            np.fill_diagonal(R_trig[k], 0.0)   # diagonal belongs to R_bg

        # ── Log-likelihood ──
        ll = (np.sum(np.log(lam_total))
              - sum(_compensator(events, mus[k], alphas[k], betas[k], T)
                    for k in range(2)))
        ll_history.append(float(ll))

        if it > 5 and (ll_history[-1] - ll_history[-2]) < tol:
            break

        # ── M-step ──
        for k in range(2):
            mus[k] = max(R_bg[k].sum() / T, 1e-8)

            p_sum = R_trig[k][lower].sum()
            w_dt  = (R_trig[k][lower] * dt[lower]).sum()

            if p_sum < 1e-10:
                continue

            def _neg_obj(x, _p=p_sum, _w=w_dt, _ev=events):
                a, b = x
                if a <= 0 or b <= 0 or a >= b:
                    return 1e10
                return -(
                    _p * np.log(a)
                    - b * _w
                    - (a / b) * np.sum(1.0 - np.exp(-b * (T - _ev)))
                )

            res = minimize(_neg_obj, [alphas[k], betas[k]],
                           method='Nelder-Mead',
                           options={'maxiter': 400, 'xatol': 1e-5,
                                    'fatol': 1e-5})
            alphas[k], betas[k] = _enforce_stability(*res.x)

    return mus, alphas, betas, R_bg, R_trig
