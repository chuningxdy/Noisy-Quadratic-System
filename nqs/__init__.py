"""
NQS (Noisy Quadratic System) package

Public API:
    compute_nqs_standard     - Standard NQS risk computation (constant/step LR schedule)
    compute_nqs_regularized  - NQS risk with weight-norm regularization
    fit_nqs                  - Fit NQS parameters to observed losses

Common input formats:

    nqs_dict : dict
        NQS model parameters. All values are float.
        {
            'p': float,      # exponent for initial risk per dimension (P/n^p)
            'q': float,      # exponent for eigenvalues (Q/n^q)
            'P': float,      # multiplier for initial risk
            'Q': float,      # multiplier for eigenvalues
            'e_irr': float,  # irreducible error (Bayes risk)
            'R': float,      # multiplier for noise variance
            'r': float,      # exponent for noise variance (R/n^r)
        }

    h : dict
        Training hyperparameters. Used by compute_nqs_standard and compute_nqs_regularized.
        {
            'N': int,        # number of model parameters (dimensionality)
            'K': int,        # number of training steps
            'B': int,        # batch size
        }

    param_ranges : dict
        Search ranges for fit_nqs. Each value is a (low, high) tuple of floats.
        Uses the same keys as nqs_dict.
        {
            'p': (float, float),      # range for p exponent
            'q': (float, float),      # range for q exponent
            'P': (float, float),      # range for P multiplier
            'Q': (float, float),      # range for Q multiplier
            'e_irr': (float, float),  # range for irreducible error
            'R': (float, float),      # range for R (noise variance multiplier)
            'r': (float, float),      # range for r exponent
        }
"""

import warnings
import jax
import jax.numpy as jnp
import time

_HAS_GPU = any(d.platform == "gpu" for d in jax.devices())

if not _HAS_GPU:
    warnings.warn(
        "No GPU detected — JAX is running on CPU. "
        "Prediction (compute_nqs_standard, compute_nqs_regularized) will work but fit_nqs requires a GPU. "
        "To enable GPU support, see: https://jax.readthedocs.io/en/latest/installation.html",
        stacklevel=2,
    )

from ._core import (
    risk as _risk,
    risk_LRA as _risk_LRA,
    _fit_nqs_internal,
    _nqs_dict_to_array,
    _nqs_array_to_dict,
    latin_hypercube_initializations,
)


_DEFAULT_LR = 1.999

def compute_nqs_standard(nqs_dict, h):
    """Compute NQS risk with a constant LR schedule.

    Args:
        nqs_dict (dict): NQS parameters {p, q, P, Q, e_irr, R, r}. All float.
        h (dict): Training hyperparameters {N: int, K: int, B: int}.

    Returns:
        float: Scalar risk value (expected loss of the NQS model).
    """
    nqs = _nqs_dict_to_array(nqs_dict)
    sch = {"decay_at": [], "decay_amt": [], "B_decay_amt": []}
    return _risk(nqs, N=h['N'], K=h['K'], B=h['B'], lr=_DEFAULT_LR, sch=sch)


def compute_nqs_regularized(nqs_dict, h, init_weight_norm_squared_fn=None):
    """Compute NQS risk with weight-norm regularization.

    Uses adaptive learning rate scaling based on weight norm tracking.

    Args:
        nqs_dict (dict): NQS parameters {p, q, P, Q, e_irr, R, r}. All float.
        h (dict): Training hyperparameters {N: int, K: int, B: int}.
        init_weight_norm_squared_fn (callable, optional): A function f(N) -> float
            that returns the initial weight norm squared for a model with N parameters.
            Default: lambda N: N * 0.02**2  (i.e., each weight initialized ~ N(0, 0.02)).

    Returns:
        float: Scalar risk value (expected loss with weight-norm regularization).
    """
    if init_weight_norm_squared_fn is None:
        init_weight_norm_squared_fn = lambda N: N * 0.02**2

    nqs = _nqs_dict_to_array(nqs_dict)
    sch = {"decay_at": [], "decay_amt": [], "B_decay_amt": []}
    return _risk_LRA(nqs, N=h['N'], K=h['K'], B=h['B'], lr=_DEFAULT_LR, sch=sch,
                     init_weight_norm_squared_fn=init_weight_norm_squared_fn)


def fit_nqs(h_dicts, nn_losses, seed, number_of_initializations, param_ranges,
            gtol, max_steps, loss='huber'):
    """Fit NQS parameters to observed neural network losses.

    Uses Adam optimizer with Latin Hypercube Sampling for initialization,
    parallelized over multiple starting points via jax.lax.scan.

    Args:
        h_dicts (list[dict]): List of training hyperparameter dicts.
            Each dict: {N: int, K: int, B: int}.
        nn_losses (list[float] or array): Observed losses, one per entry in h_dicts.
        seed (int): Random seed for Latin Hypercube Sampling initialization.
        number_of_initializations (int): Number of random starting points for optimization.
        param_ranges (dict): Search ranges for each NQS parameter.
            Keys: 'p', 'q', 'P', 'Q', 'e_irr', 'R', 'r'.
            Each value is a (low: float, high: float) tuple.
        gtol (float): Gradient norm convergence tolerance.
        max_steps (int): Maximum number of Adam iterations.
        loss (str): Loss function for fitting. 'huber' (default) or 'mse'.
            'huber' uses Huber loss in log-space (delta=1e-3).
            'mse' uses squared log-error: 0.5 * (log y - log y_hat)^2.

    Returns:
        tuple: (fitted_nqs_dict, eval_metric_value, None)
            - fitted_nqs_dict (dict): Fitted NQS parameters {p, q, P, Q, e_irr, R, r}.
              All values are float.
            - eval_metric_value (float): RMSLE percentage metric = sqrt(2 * loss) * 100.
    """
    if not _HAS_GPU:
        raise RuntimeError(
            "fit_nqs requires a GPU but none was detected. "
            "Install a CUDA-enabled JAX: https://jax.readthedocs.io/en/latest/installation.html"
        )

    list_of_nqs_inits = latin_hypercube_initializations(
        seed=seed,
        num_inits=number_of_initializations,
        param_names=['p', 'q', 'P', 'Q', 'e_irr', 'R', 'r'],
        param_ranges={
            'p': (param_ranges['p'][0], param_ranges['p'][1]),
            'q': (param_ranges['q'][0], param_ranges['q'][1]),
            'P': (param_ranges['P'][0], param_ranges['P'][1]),
            'Q': (param_ranges['Q'][0], param_ranges['Q'][1]),
            'e_irr': (param_ranges['e_irr'][0], param_ranges['e_irr'][1]),
            'R': (param_ranges['R'][0], param_ranges['R'][1]),
            'r': (param_ranges['r'][0], param_ranges['r'][1])
        },
        r_equals_q=False)

    cfg_arrs = []
    for h_dict in h_dicts:
        cfg_arrs.append(jnp.array([h_dict["N"], h_dict["K"], h_dict["B"], _DEFAULT_LR]))

    ys = jnp.array(nn_losses)
    cfgs = jnp.array(cfg_arrs)

    if loss == 'huber':
        use_huber = True
    elif loss == 'mse':
        use_huber = False
    else:
        raise ValueError("loss must be 'huber' or 'mse'")

    start_time = time.time()
    best_nqs, best_loss, best_idx, trajectories = _fit_nqs_internal(
        list_of_nqs_inits,
        cfgs,
        ys,
        itrs=max_steps,
        return_trajectories=True,
        tie_r_and_q=False,
        use_huber=use_huber)
    end_time = time.time()

    print(f"fit_nqs completed in {end_time - start_time:.2f}s | best_loss={best_loss:.6f}")

    fitted_nqs_dict = _nqs_array_to_dict(best_nqs)

    fit_metric_value = jnp.array(best_loss)
    eval_metric_value = jnp.sqrt(2 * fit_metric_value) * 100

    return fitted_nqs_dict, eval_metric_value, None
