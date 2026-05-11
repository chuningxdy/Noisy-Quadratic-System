"""
Internal computation machinery for the NQS package.

NQS (Noisy Quadratic System) model:
  Q(w) = e + 1/2 sum_{n=1}^inf lambda_n (w_n - w_n^*)^2
  lambda_n = Q/n^q
  E[lambda_n * (w_n^(0) - w^*)^2] = P/n^p

Expected risk decomposes as:
  E[Q(w^(K))] = e_irr + e_appx + e_est
  where e_est = e_bias + e_var
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
if not (jnp.array(1.).dtype is jnp.dtype("float64")):
    raise RuntimeError("jax.config.jax_enable_x64 must be set for numerical stability.")
from collections import namedtuple
from jax import lax

from scipy.stats import qmc


# -------------------------------------------------------- #
#                    Types & Constants                     #
# -------------------------------------------------------- #

Cfg = namedtuple("Cfg", ["N", "B", "K", "lr", "sch"])
NQS_DIM = 7  # number of NQS parameters: p, q, P, Q, e_irr, R, r


# -------------------------------------------------------- #
#                  Schedule Processing                      #
# -------------------------------------------------------- #

def _process_schedule_steps(lr, B, K, sch):
    """Return array of (lr, B, num_steps) rows representing schedule segments."""
    init_lr = lr
    init_B = B

    steps = []
    prev_step = 0
    for decay_at, decay_amt, B_decay_amt in zip(sch["decay_at"], sch["decay_amt"], sch["B_decay_amt"]):
        decay_step = int(decay_at * K)
        num_steps = decay_step - prev_step
        steps.append(jnp.array([lr, B, num_steps]))
        lr = init_lr * decay_amt
        B = init_B * B_decay_amt
        prev_step = decay_step

    num_steps = K - prev_step
    steps.append(jnp.array([lr, B, num_steps]))

    return jnp.array(steps)


def _merge_schedules(sch, change_points):
    """Merge a schedule with additional change points."""
    sch_decay_ats = set(sch['decay_at'])
    additional_change_points = [item for item in change_points if item not in sch_decay_ats]
    sch_labeled = [{**{key: sch[key][j] for key in sch}, 'sch_id': 1} for j in range(len(sch['decay_at']))]
    additional_change_points_labeled = [{'decay_at': item, 'sch_id': 2} for item in additional_change_points]
    merged_sch = sorted(sch_labeled + additional_change_points_labeled, key=lambda x: x['decay_at'])
    for i in range(len(merged_sch)):
        if 'decay_amt' not in merged_sch[i]:
            merged_sch[i]['decay_amt'] = 1.0 if i == 0 else merged_sch[i-1]['decay_amt']
        if 'B_decay_amt' not in merged_sch[i]:
            merged_sch[i]['B_decay_amt'] = 1.0 if i == 0 else merged_sch[i-1]['B_decay_amt']
    merged_sch_dict = {'decay_at': [], 'decay_amt': [], 'B_decay_amt': [], 'sch_id': []}
    for item in merged_sch:
        merged_sch_dict['decay_at'].append(item['decay_at'])
        merged_sch_dict['decay_amt'].append(item['decay_amt'])
        merged_sch_dict['B_decay_amt'].append(item['B_decay_amt'])
        merged_sch_dict['sch_id'].append(item['sch_id'])
    return merged_sch_dict


def _process_schedule_steps_LRA(lr, B, K, sch, interval=1000):
    """Return schedule segments with change-point labels for LR adaptation."""
    num_changes = K // interval
    change_points = [(i+1) * interval / K for i in range(num_changes)]
    sch_with_chg_pts = _merge_schedules(sch, change_points)

    init_lr = lr
    init_B = B

    steps = []
    prev_step = 0
    prev_decay_at = 0.0
    for decay_at, decay_amt, B_decay_amt, sch_id in zip(
            sch_with_chg_pts['decay_at'], sch_with_chg_pts['decay_amt'],
            sch_with_chg_pts['B_decay_amt'], sch_with_chg_pts['sch_id']):
        decay_step = int(decay_at * K)
        num_steps = decay_step - prev_step
        is_cp = 1 if prev_decay_at in change_points else 0
        steps.append(jnp.array([lr, B, num_steps, is_cp]))
        lr = init_lr * decay_amt
        B = init_B * B_decay_amt
        prev_step = decay_step
        prev_decay_at = decay_at

    num_steps = K - prev_step
    is_cp = 1 if prev_decay_at in change_points else 0
    steps.append(jnp.array([lr, B, num_steps, is_cp]))

    return jnp.array(steps)


# -------------------------------------------------------- #
#          Basic Risk Components (e_irr, e_appx)           #
# -------------------------------------------------------- #

def _e_irr(nqs, cfg):
    p, q, P, Q, e_irr, R, r = nqs
    return e_irr

def _e_irr_no_sch(nqs, cfg_array):
    p, q, P, Q, e_irr, R, r = nqs
    return e_irr

def _e_appx(nqs, cfg):
    N = cfg.N
    p, q, P, Q, e_irr, R, r = nqs
    return 0.5 * P * jnp.squeeze(jax.scipy.special.zeta(p, N+1))

def _e_appx_no_sch(nqs, cfg_array):
    N = cfg_array[0]
    p, q, P, Q, e_irr, R, r = nqs
    return 0.5 * P * jnp.squeeze(jax.scipy.special.zeta(p, N+1))


# -------------------------------------------------------- #
#            Numerical Integration (Quadrature + EM)       #
# -------------------------------------------------------- #

_GAUSS_LEGENDRE_20_X, _GAUSS_LEGENDRE_20_W = jnp.polynomial.legendre.leggauss(20)
_GAUSS_LEGENDRE_20_X, _GAUSS_LEGENDRE_20_W = jnp.array(_GAUSS_LEGENDRE_20_X), jnp.array(_GAUSS_LEGENDRE_20_W)

def _jax_quad(f, a, b):
    """Fixed 20-point Gauss-Legendre quadrature, vectorized over multiple integrals."""
    mid = (b + a) / 2
    radius = (b - a) / 2
    mapped_x = mid + radius * _GAUSS_LEGENDRE_20_X
    y_values = jax.vmap(f)(mapped_x)
    result = radius * jnp.sum(_GAUSS_LEGENDRE_20_W[:, jnp.newaxis] * y_values, axis=0)
    return result


def _em(g, L, U):
    """First-order Euler-Maclaurin approximation for sum g(L+1) + ... + g(U)."""
    def f(x):
        n = jnp.exp(x)
        dn = jnp.exp(x)
        return g(n) * dn

    logL = jnp.log(L)
    logU = jnp.log(U)
    integral = _jax_quad(f, logL, logU)
    risk = integral + (g(U) - g(L)) / 2
    return risk


# -------------------------------------------------------- #
#                  Recursive Reduction                      #
# -------------------------------------------------------- #

def _reduce(f, input_dim=1):
    """Recursive reduction for summation over n=1 to N."""
    def reducedf(N):
        def bodyf(val):
            s, n = val
            return (s + f(n), n-1)

        def condf(val):
            s, n = val
            return n > 0

        risk, _ = jax.lax.while_loop(condf, bodyf, (jnp.array([0.0]*input_dim), N))
        return risk

    return reducedf


# -------------------------------------------------------- #
#                    Geometric Sums                         #
# -------------------------------------------------------- #

def _geom_sum(a, n, output_prod=False):
    """Return S = 1 + a + a^2 + ... + a^(n-1)."""
    ans_pow = a ** n
    ans_sum = (1 - ans_pow) / (1 - a)
    return (ans_sum, ans_pow) if output_prod else ans_sum


# -------------------------------------------------------- #
#         Fast Computation of Bias & Variance               #
# -------------------------------------------------------- #

def _e_dim_bv_steps(nqs, n, steps):
    """Accumulate bias/variance factors at dimension n over schedule phases."""
    b_factor = 1.0
    v_factor = 0.0
    p, q, P, Q, e_irr, R, r = nqs

    for step in steps:
        lr = step[0]
        B = step[1]
        num_steps = step[2]
        a = 1.0 - lr * (Q / n**q)
        sumprod_factor, prod_factor = _geom_sum(a**2, num_steps, output_prod=True)
        b_factor = b_factor * prod_factor
        v_factor = v_factor * prod_factor + lr**2/B * sumprod_factor

    return b_factor, v_factor


def _f(nqs, n, steps):
    """Per-dimension bias and variance contribution."""
    p, q, P, Q, e_irr, R, r = nqs
    init_R_n = 0.5 * P / n**p
    lambda_n = Q / n**q
    b_factor, v_factor = _e_dim_bv_steps(nqs, n, steps)
    e_bias_n = init_R_n * b_factor
    e_var_n = 0.5 * lambda_n * (R / n**r) * v_factor
    return jnp.array([e_bias_n, e_var_n])


# JIT-compiled reduce and EM for _f
_jit_reduce_f = jax.jit(lambda nqs, N, steps: _reduce(lambda n: _f(nqs, n, steps), input_dim=2)(N))
_jit_em_f = jax.jit(lambda L, U, nqs, steps: _em(lambda n: _f(nqs, n, steps), L, U))


def _e_bias_var_fast(nqs, cfg):
    """Fast computation of e_bias and e_var using exact sum + EM approximation."""
    N = cfg.N
    K = cfg.K
    B = cfg.B
    lr = cfg.lr
    sch = cfg.sch
    steps = _process_schedule_steps(lr, B, K, sch)

    M = jnp.minimum(jnp.maximum(1, jnp.array((N * 0.05), int)), 100)
    e_bv_1_to_M = _jit_reduce_f(nqs, M, steps)
    e_bv_Mplus1_to_N = _jit_em_f(M, N, nqs, steps)
    return e_bv_1_to_M + e_bv_Mplus1_to_N


def _e_bias_var_fast_no_sch(nqs, cfg_array):
    """Fast computation of e_bias and e_var (no schedule variant)."""
    N = cfg_array[0]
    K = cfg_array[1]
    B = cfg_array[2]
    lr = cfg_array[3]
    steps = _process_schedule_steps(lr, B, K, {"decay_at": [], "decay_amt": [], "B_decay_amt": []})

    M = jnp.minimum(jnp.maximum(1, jnp.array((N * 0.05), int)), 100)
    e_bv_1_to_M = _jit_reduce_f(nqs, M, steps)
    e_bv_Mplus1_to_N = _jit_em_f(M, N, nqs, steps)
    return e_bv_1_to_M + e_bv_Mplus1_to_N


# -------------------------------------------------------- #
#                     Risk Functions                        #
# -------------------------------------------------------- #

def _risk(nqs, cfg):
    e_est_bv = _e_bias_var_fast(nqs, cfg)
    e_appx = _e_appx(nqs, cfg)
    e_irr = _e_irr(nqs, cfg)
    return e_est_bv[0] + e_est_bv[1] + e_appx + e_irr


def risk(nqs, N, K, B, lr, sch):
    """Compute standard NQS risk from raw arguments."""
    return _risk(nqs, Cfg(N=N, K=K, B=B, lr=lr, sch=sch))


def _risk_no_sch(nqs, cfg_array):
    e_est_bv = _e_bias_var_fast_no_sch(nqs, cfg_array)
    e_appx = _e_appx_no_sch(nqs, cfg_array)
    e_irr = _e_irr_no_sch(nqs, cfg_array)
    return e_est_bv[0] + e_est_bv[1] + e_appx + e_irr


# -------------------------------------------------------- #
#                    Gradient Computation                   #
# -------------------------------------------------------- #

@jax.jit
def _grad_e_irr_no_sch(nqs, cfg_array):
    return jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

@jax.jit
def _grad_e_appx_no_sch(nqs, cfg_array):
    return jax.grad(lambda nqs_local: _e_appx_no_sch(nqs_local, cfg_array))(nqs)


def _g(nqs, n, steps):
    """Scalar bias+variance at dimension n (for gradient computation)."""
    p, q, P, Q, e_irr, R, r = nqs
    init_R_n = 0.5 * P / n**p
    lambda_n = Q / n**q
    b_factor, v_factor = _e_dim_bv_steps(nqs, n, steps)
    e_bias_n = init_R_n * b_factor
    e_var_n = 0.5 * lambda_n * (R / n**r) * v_factor
    return jnp.squeeze(e_bias_n + e_var_n)


@jax.jit
def _grad_g(nqs, n, steps):
    return jax.grad(lambda nqs_local: _g(nqs_local, n, steps))(nqs)


# JIT-compiled reduce and EM for gradients
_reduced_grad_g = jax.jit(
    lambda nqs, N, steps: _reduce(lambda n: _grad_g(nqs, n, steps), input_dim=NQS_DIM)(N))
_em_grad_g = jax.jit(
    lambda L, U, nqs, steps: _em(lambda n: _grad_g(nqs, n, steps), L, U))


def _grad_e_bias_var_fast_no_sch(nqs, cfg_array):
    """Fast gradient of e_bias+e_var w.r.t. NQS params (no schedule)."""
    N = cfg_array[0]
    K = cfg_array[1]
    B = cfg_array[2]
    lr = cfg_array[3]
    steps = _process_schedule_steps(lr, B, K, {"decay_at": [], "decay_amt": [], "B_decay_amt": []})

    M = jnp.minimum(jnp.maximum(1, jnp.array((N * 0.05), int)), 100)
    e_grad_bv_1_to_M = _reduced_grad_g(nqs, M, steps)
    e_grad_bv_Mplus1_to_N = _em_grad_g(M, N, nqs, steps)
    return e_grad_bv_1_to_M + e_grad_bv_Mplus1_to_N


def _grad_risk_no_sch(nqs, cfg_array):
    return (_grad_e_bias_var_fast_no_sch(nqs, cfg_array) +
            _grad_e_appx_no_sch(nqs, cfg_array) +
            _grad_e_irr_no_sch(nqs, cfg_array))


# -------------------------------------------------------- #
#       Spectral-Norm Regularized Bias/Variance             #
# -------------------------------------------------------- #

def _e_dim_bv_one_step(nqs, n, bv_factor, step, b_decay_factor=1.0):
    """Advance bias/variance factors for one schedule phase."""
    p, q, P, Q, e_irr, R, r = nqs
    b_factor, v_factor = bv_factor

    lr = step[0]
    B = step[1]
    num_steps = step[2]
    a = 1.0 - lr * (Q / n**q)

    sumprod_factor, prod_factor = _geom_sum(a**2, num_steps, output_prod=True)

    b_factor = b_factor * prod_factor
    v_factor = v_factor * prod_factor + lr**2/B * sumprod_factor * b_decay_factor
    bv_factor = jnp.array([b_factor, v_factor])

    return bv_factor


def _get_em_quadrature_points(L, U):
    logL = jnp.log(L)
    logU = jnp.log(U)
    mid = (logU + logL) / 2
    radius = (logU - logL) / 2
    mapped_x = mid + radius * _GAUSS_LEGENDRE_20_X
    mapped_n = jnp.exp(mapped_x)
    return mapped_n, _GAUSS_LEGENDRE_20_W


def _init_all_points(L, U, include_1_to_L=True):
    """Initialize quadrature + integer points with bias/variance factors."""
    quad_n, quad_w = _get_em_quadrature_points(L, U)
    if include_1_to_L:
        one_to_L = jnp.arange(1, L+1)
        all_n = jnp.concatenate([quad_n, one_to_L])
        radius = (jnp.log(U) - jnp.log(L)) / 2
        all_w = jnp.concatenate([quad_w * radius, jnp.ones_like(one_to_L)])
    else:
        all_n = quad_n
        radius = (jnp.log(U) - jnp.log(L)) / 2
        all_w = quad_w * radius

    bv_factors = jnp.tile(jnp.array([1.0, 0.0]), (all_n.shape[0], 1))
    start_factors = jnp.concatenate([all_n[:, None], all_w[:, None], bv_factors], axis=1)
    return start_factors


def _bv_from_bv_factor(nqs, n, bv_factor):
    p, q, P, Q, e_irr, R, r = nqs
    b_factor, v_factor = bv_factor
    init_R_n = 0.5 * P / n**p
    lambda_n = Q / n**q
    e_bias_n = init_R_n * b_factor
    e_var_n = 0.5 * lambda_n * (R / n**r) * v_factor
    return jnp.array([e_bias_n, e_var_n])


def _w2_from_bv_factor(nqs, n, bv_factor):
    p, q, P, Q, e_irr, R, r = nqs
    b_factor, v_factor = bv_factor
    init_R_n = 0.5 * P / n**p
    lambda_n = Q / n**q
    e_bias_n = init_R_n * b_factor
    e_var_n = 0.5 * lambda_n * (R / n**r) * v_factor
    w_star2_n = P / n**p / lambda_n
    e_w2_n = (1 - 2 * jnp.sqrt(b_factor)) * w_star2_n + 2 / lambda_n * (e_bias_n + e_var_n)
    return e_w2_n


@jax.jit
def _em_step(start_factors, nqs, step, L, b_decay_factor=1.0):
    """Advance all quadrature+integer points by one schedule step.

    Args:
        start_factors: array of shape (num_points, 4) with columns [n, w, b_factor, v_factor]
        nqs: NQS parameter array
        step: schedule step array [lr, B, num_steps, ...]
        L: boundary between integer-sum points (n <= L) and quadrature points (n > L).
           Points with n > L get multiplied by the Jacobian factor n (for change of variable).
        b_decay_factor: noise scaling factor (for regularization)

    Returns:
        (bias_var, w2, end_factors)
    """
    quadrature_points = start_factors[:, 0]
    w = start_factors[:, 1]
    bv_factors_curr = start_factors[:, 2:4]

    bv_factors_stepped = jax.vmap(
        lambda n, bv_factor: _e_dim_bv_one_step(nqs, n, bv_factor, step, b_decay_factor)
    )(quadrature_points, bv_factors_curr)
    bv_values = jax.vmap(
        lambda n, bv_factor: _bv_from_bv_factor(nqs, n, bv_factor)
    )(quadrature_points, bv_factors_stepped)
    w2_values = jax.vmap(
        lambda n, bv_factor: _w2_from_bv_factor(nqs, n, bv_factor)
    )(quadrature_points, bv_factors_stepped)

    # Quadrature points (n > L) need Jacobian factor; integer points (n <= L) get 1.0
    y_values = bv_values * jnp.where(quadrature_points[:, None] > L, quadrature_points[:, None], 1.0)
    w2_values = w2_values * jnp.where(quadrature_points > L, quadrature_points, 1.0)

    result = jnp.sum(w[:, jnp.newaxis] * y_values, axis=0)
    result_w2 = jnp.sum(w * w2_values, axis=0)

    end_factors = jnp.concatenate([quadrature_points[:, None], w[:, None], bv_factors_stepped], axis=1)

    return result, result_w2, end_factors


@jax.jit
def _em_at(start_factors, nqs, L):
    """Evaluate bias/variance and w2 at current state (no stepping).

    Args:
        start_factors: array of shape (num_points, 4)
        nqs: NQS parameter array
        L: boundary between integer-sum points and quadrature points
    """
    quadrature_points = start_factors[:, 0]
    w = start_factors[:, 1]
    bv_factors_curr = start_factors[:, 2:4]

    bv_values = jax.vmap(
        lambda n, bv_factor: _bv_from_bv_factor(nqs, n, bv_factor)
    )(quadrature_points, bv_factors_curr)
    w2_values = jax.vmap(
        lambda n, bv_factor: _w2_from_bv_factor(nqs, n, bv_factor)
    )(quadrature_points, bv_factors_curr)

    y_values = bv_values * jnp.where(quadrature_points[:, None] > L, quadrature_points[:, None], 1.0)
    w2_values = w2_values * jnp.where(quadrature_points > L, quadrature_points, 1.0)

    result = jnp.sum(w[:, jnp.newaxis] * y_values, axis=0)
    result_w2 = jnp.sum(w * w2_values, axis=0)

    return result, result_w2


def _e_bias_var_SN_fast(nqs, cfg, interval=1000, init_weight_norm_squared_fn=None):
    """Fast computation of e_bias and e_var with weight norm regularization."""
    N = cfg.N
    M = jnp.minimum(jnp.maximum(1, jnp.array((N * 0.05), int)), 100)

    K_all = cfg.K
    B = cfg.B
    lr = cfg.lr
    sch = cfg.sch
    steps_all = _process_schedule_steps_LRA(lr, B, K_all, sch, interval=interval)

    init_weight_norm_squared = init_weight_norm_squared_fn(N)

    K = 0
    init_lr = cfg.lr
    curr_lr = cfg.lr

    for i in range(0, steps_all.shape[0]):
        if i == 0:
            start_factors = _init_all_points(M, N, include_1_to_L=True)
            bias_var_init, w2_init = _em_at(start_factors, nqs, M)
            loss = bias_var_init[0] + bias_var_init[1]
            w2 = w2_init

        noise_scale = 1.0
        if w2 < 0.0:
            lr_scale = 1.0
        else:
            lr_scale = 1 - w2/(init_weight_norm_squared + w2)

        step = steps_all[i, :]
        step_lr = step[0]
        step_lr = jnp.minimum(curr_lr, jnp.minimum(step_lr, init_lr * lr_scale))
        curr_lr = step_lr
        steps_all = steps_all.at[i, 0].set(step_lr)
        step = steps_all[i, :]

        bias_var_step, w2_step, end_factors = _em_step(
            start_factors, nqs, step, M, b_decay_factor=noise_scale)
        e_bv_at_step = bias_var_step

        K = K + step[2]
        start_factors = end_factors
        loss = e_bv_at_step[0] + e_bv_at_step[1]
        w2 = w2_step

    return e_bv_at_step, w2


def _risk_LRA(nqs, cfg, init_weight_norm_squared_fn=None):
    K_target = cfg.K
    if K_target > 10:
        stair_width = max(K_target // 100, min(1000, K_target // 10))
    else:
        stair_width = 1

    e_est_bv, _ = _e_bias_var_SN_fast(nqs, cfg, interval=stair_width,
                                        init_weight_norm_squared_fn=init_weight_norm_squared_fn)
    e_appx = _e_appx(nqs, cfg)
    e_irr_val = _e_irr(nqs, cfg)
    return e_est_bv[0] + e_est_bv[1] + e_appx + e_irr_val


def risk_LRA(nqs, N, K, B, lr, sch, init_weight_norm_squared_fn=None):
    """Compute regularized NQS risk from raw arguments."""
    return _risk_LRA(nqs, Cfg(N=N, K=K, B=B, lr=lr, sch=sch),
                     init_weight_norm_squared_fn=init_weight_norm_squared_fn)


# -------------------------------------------------------- #
#          Parameter Transformations for Fitting            #
# -------------------------------------------------------- #

def _nqs_to_x(nqs, norm_const=1e5):
    """Convert NQS params to normalized x params for optimization."""
    log_norm_const = jnp.log(norm_const)
    x = jnp.zeros_like(nqs)
    x = x.at[0].set(nqs[0])                              # p
    x = x.at[1].set(nqs[1])                              # q
    x = x.at[2].set(jnp.log(nqs[2]) / log_norm_const)   # P
    x = x.at[3].set(jnp.log(nqs[3]) / log_norm_const)   # Q
    x = x.at[4].set(jnp.log(nqs[4]) / log_norm_const)   # e_irr
    x = x.at[5].set(jnp.log(nqs[5]) / log_norm_const)   # R
    x = x.at[6].set(nqs[6])                              # r
    return x


def _x_to_nqs(x, norm_const=1e5):
    """Convert normalized x params back to NQS params."""
    nqs = jnp.zeros_like(x)
    nqs = nqs.at[0].set(x[0])                # p
    nqs = nqs.at[1].set(x[1])                # q
    nqs = nqs.at[2].set(norm_const ** x[2])   # P
    nqs = nqs.at[3].set(norm_const ** x[3])   # Q
    nqs = nqs.at[4].set(norm_const ** x[4])   # e_irr
    nqs = nqs.at[5].set(norm_const ** x[5])   # R
    nqs = nqs.at[6].set(x[6])                # r
    return nqs


def _dnqs_dx(x, nqs, norm_const=1e5):
    """Diagonal Jacobian of nqs w.r.t. x."""
    log_norm_const = jnp.log(norm_const)
    dnqs_dx = jnp.zeros_like(x)
    dnqs_dx = dnqs_dx.at[0].set(1.0)
    dnqs_dx = dnqs_dx.at[1].set(1.0)
    dnqs_dx = dnqs_dx.at[2].set(log_norm_const * nqs[2])
    dnqs_dx = dnqs_dx.at[3].set(log_norm_const * nqs[3])
    dnqs_dx = dnqs_dx.at[4].set(log_norm_const * nqs[4])
    dnqs_dx = dnqs_dx.at[5].set(log_norm_const * nqs[5])
    dnqs_dx = dnqs_dx.at[6].set(1.0)
    return dnqs_dx


# -------------------------------------------------------- #
#                  Loss & Gradient for Fitting              #
# -------------------------------------------------------- #

HUBER_DELTA = 1e-3

def _loss_mse(y_hat, y):
    """Squared log-error loss: 0.5 * (log y - log y_hat)^2."""
    return 0.5 * (jnp.log(y) - jnp.log(y_hat)) ** 2

def _loss_huber(y_hat, y):
    """Huber loss in log-space with delta=HUBER_DELTA."""
    log_diff = jnp.log(y) - jnp.log(y_hat)
    abs_diff = jnp.abs(log_diff)
    raw = 0.5 * log_diff ** 2
    huber = HUBER_DELTA * (abs_diff - 0.5 * HUBER_DELTA)
    return jnp.where(abs_diff <= HUBER_DELTA, raw, huber)

dloss_mse = jax.jit(jax.grad(_loss_mse, argnums=0))
dloss_huber = jax.jit(jax.grad(_loss_huber, argnums=0))


def _grad_loss_no_sch(nqs, y, cfg_array, tie_r_and_q=True, dloss_fn=dloss_mse):
    y_hat = _risk_no_sch(nqs, cfg_array)
    drisk = _grad_risk_no_sch(nqs, cfg_array)
    if tie_r_and_q:
        drisk = drisk.at[6].add(drisk[1])
        drisk = drisk.at[1].set(drisk[6])
    grad_loss_nqs = dloss_fn(y_hat, y) * drisk
    return grad_loss_nqs


def _grad_loss_normalized_no_sch(x, y, cfg_array, tie_r_and_q=True, dloss_fn=dloss_mse):
    nqs = _x_to_nqs(x)
    grad_loss_nqs = _grad_loss_no_sch(nqs, y, cfg_array, tie_r_and_q=tie_r_and_q, dloss_fn=dloss_fn)
    dnqs_dx_val = _dnqs_dx(x, nqs)
    return grad_loss_nqs * dnqs_dx_val


def _make_grad_total_loss(tie_r_and_q, dloss_fn):
    return lambda x, ys, cfg_arrays: jnp.sum(
        jax.vmap(lambda x_local, y_local, cfg_local: _grad_loss_normalized_no_sch(
            x_local, y_local, cfg_local, tie_r_and_q=tie_r_and_q, dloss_fn=dloss_fn),
                 in_axes=(None, 0, 0))(x, ys, cfg_arrays), axis=0)


# -------------------------------------------------------- #
#                  LHS Initialization                       #
# -------------------------------------------------------- #

def latin_hypercube_initializations(seed, num_inits, param_names, param_ranges, r_equals_q=True):
    """Create initialization points using Latin Hypercube Sampling."""
    lower_bounds = [param_range[0] for param_range in param_ranges.values()]
    upper_bounds = [param_range[1] for param_range in param_ranges.values()]

    samples = qmc.LatinHypercube(d=len(param_names), seed=seed).random(n=num_inits)

    init_nqs_list = []
    for i in range(num_inits):
        sample = samples[i]
        param_values = jnp.array([0.0] * len(param_names))
        for j in range(len(param_names)):
            param_values = param_values.at[j].set(lower_bounds[j] + sample[j] * (upper_bounds[j] - lower_bounds[j]))
        init_nqs_list.append(param_values)

    init_nqs_array = jnp.array(init_nqs_list)

    r_index = param_names.index('r')
    q_index = param_names.index('q')
    if r_equals_q:
        init_nqs_array = init_nqs_array.at[:, r_index].set(init_nqs_array[:, q_index])

    return init_nqs_array


# -------------------------------------------------------- #
#                    NQS Array <-> Dict                     #
# -------------------------------------------------------- #

def _nqs_dict_to_array(nqs_dict):
    """Convert an NQS dict to the internal [p, q, P, Q, e_irr, R, r] array."""
    return jnp.array([
        nqs_dict['p'],
        nqs_dict['q'],
        nqs_dict['P'],
        nqs_dict['Q'],
        nqs_dict['e_irr'],
        nqs_dict['R'],
        nqs_dict['r'],
    ])


def _nqs_array_to_dict(nqs_array):
    """Convert an internal NQS array back to a dict."""
    return {
        'p': float(nqs_array[0]),
        'q': float(nqs_array[1]),
        'P': float(nqs_array[2]),
        'Q': float(nqs_array[3]),
        'e_irr': float(nqs_array[4]),
        'R': float(nqs_array[5]),
        'r': float(nqs_array[6]),
    }


# -------------------------------------------------------- #
#                 NQS Fitting (Adam)                        #
# -------------------------------------------------------- #

def _fit_nqs_internal(list_of_nqs_inits, cfg_arrays, ys, itrs=3, gtol=1e-8,
                      return_trajectories=False, tie_r_and_q=True, use_huber=False):
    """Fit NQS parameters using Adam optimizer with multiple initializations."""

    _loss_fn = _loss_huber if use_huber else _loss_mse
    _dloss_fn = dloss_huber if use_huber else dloss_mse

    @jax.jit
    def _calc_total_loss_no_sch(x):
        nqs = _x_to_nqs(x)
        total_loss = 0.0
        for i in range(len(ys)):
            y = ys[i]
            cfg = cfg_arrays[i]
            loss_i = _loss_fn(_risk_no_sch(nqs, cfg), y)
            total_loss = total_loss + loss_i
        return total_loss

    _grad_total_loss = _make_grad_total_loss(tie_r_and_q=tie_r_and_q, dloss_fn=_dloss_fn)

    @jax.jit
    def _calc_grad(x):
        return _grad_total_loss(x, ys, cfg_arrays)

    def _step_fn(carry, step_idx):
        x, m, v, active_flag = carry

        learning_rate = 0.001
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        grad = _calc_grad(x)

        clip_at = 1.0
        clipped_grad = jnp.clip(grad, -clip_at, clip_at)

        m_new = jnp.where(active_flag, beta1 * m + (1 - beta1) * clipped_grad, m)
        v_new = jnp.where(active_flag, beta2 * v + (1 - beta2) * (clipped_grad ** 2), v)

        m_hat = m_new / (1 - beta1 ** (step_idx + 1))
        v_hat = v_new / (1 - beta2 ** (step_idx + 1))

        update = learning_rate * m_hat / (jnp.sqrt(v_hat) + epsilon)
        x_new = jnp.where(active_flag, x - update, x)

        grad_norm = jnp.linalg.norm(grad)
        converged_grad = grad_norm < gtol
        new_active_flag = active_flag & ~(converged_grad)

        traj_info = (x_new, grad, grad_norm, new_active_flag)
        return (x_new, m_new, v_new, new_active_flag), traj_info

    @jax.jit
    def _batch_step_fn(batch_state, step_idx):
        return jax.vmap(lambda state: _step_fn(state, step_idx))(batch_state)

    list_of_x_inits = [_nqs_to_x(nqs_init) for nqs_init in list_of_nqs_inits]
    init_x = jnp.stack(list_of_x_inits)
    init_m = jnp.zeros_like(init_x)
    init_v = jnp.zeros_like(init_x)
    init_active_flags = jnp.ones(len(list_of_x_inits), dtype=bool)

    init_state = (init_x, init_m, init_v, init_active_flags)

    @jax.jit
    def scan_step(state_and_traj, i):
        state, traj_list = state_and_traj
        new_state, new_traj_info = _batch_step_fn(state, i)
        return (new_state, traj_list), new_traj_info

    (final_state, _), traj_infos = jax.lax.scan(
        scan_step,
        (init_state, []),
        jnp.arange(itrs)
    )

    final_nqs_params, _, _, _ = final_state
    final_losses = jax.vmap(lambda x: _calc_total_loss_no_sch(x))(final_nqs_params)
    best_idx = jnp.nanargmin(final_losses)
    best_x = final_nqs_params[best_idx]
    best_nqs = _x_to_nqs(best_x)
    best_loss = final_losses[best_idx]

    trajectories = list(traj_infos)

    if return_trajectories:
        return best_nqs, best_loss, best_idx, trajectories
    else:
        return best_nqs, best_loss
