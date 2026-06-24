# NQS: Predicting Large Model Test Losses with a Noisy Quadratic System

This is the accompanying code for the paper [Predicting Large Model Test Losses with a Noisy Quadratic System](https://arxiv.org/abs/2605.09154v1).

This package implements the Noisy Quadratic System (NQS) model for predicting the pre-training loss of large language models (LLMs). Like Chinchilla scaling laws, NQS must first be fitted to a set of observed training runs (varying model size, batch size, and training steps) to learn the scaling parameters for a given model family and dataset. Once fitted, it predicts the expected training loss for any combination of these hyperparameters without running the actual training. This enables practitioners to explore scaling trade-offs, plan compute-optimal training configurations, and estimate model performance across a wide range of scales before committing GPU resources.

## Installation

```bash
pip install .                # CPU-only
pip install ".[cuda]"        # with GPU support (CUDA 12)
```

This installs the package with JAX (CPU-only by default), NumPy, and SciPy.

**GPU requirement for fitting:** `fit_nqs` requires a GPU and will raise a `RuntimeError` if none is detected. Prediction functions (`compute_nqs_standard`, `compute_nqs_regularized`) work on CPU, though you will see a warning at import time:

```
UserWarning: No GPU detected — JAX is running on CPU.
Prediction (compute_nqs_standard, compute_nqs_regularized) will work but fit_nqs requires a GPU.
```

To enable GPU support, install with the `cuda` extra:
```bash
pip install ".[cuda]"
```

For other CUDA versions or platforms, see the JAX installation guide:
https://jax.readthedocs.io/en/latest/installation.html

## Package API

The `nqs` package exposes three functions:

### `compute_nqs_standard(nqs_dict, h)`

Compute NQS risk with a constant learning rate schedule.

```python
from nqs import compute_nqs_standard

nqs_params = {
    'p': 1.117, 'q': 0.588, 'P': 3.581, 'Q': 0.936,
    'e_irr': 0.449, 'R': 4.283, 'r': 1.485,
}

h = {'N': 64_000_000, 'K': 10_000, 'B': 2048}

predicted_loss = compute_nqs_standard(nqs_params, h)
```

### `compute_nqs_regularized(nqs_dict, h, init_weight_norm_squared_fn=None)`

Compute NQS risk with weight-norm regularization and adaptive learning rate scaling.

```python
from nqs import compute_nqs_regularized

# Custom function: initial weight norm squared as a function of model size (N)
# (default: lambda N: N * 0.02**2)
init_fn = lambda N: N * 0.02**2

predicted_loss = compute_nqs_regularized(nqs_params, h, init_weight_norm_squared_fn=init_fn)
```

### `fit_nqs(h_dicts, nn_losses, seed, number_of_initializations, param_ranges, gtol, max_steps, loss='huber')`

Fit NQS parameters to observed neural network losses. Uses Adam with Latin Hypercube Sampling for multi-start optimization.

```python
from nqs import fit_nqs

h_dicts = [
    {'N': 64_000_000, 'K': 10_000, 'B': 2048},
    {'N': 128_000_000, 'K': 5_000, 'B': 4096},
    # ...
]
nn_losses = [2.41, 2.16, ...]

# custom initialisation range for the fitting procedure
# the initial values of p, q, P, Q, ...are randomly sampled from the ranges below
param_ranges = {
    'p': (1.05, 2.5),
    'q': (0.6, 2.5),
    'P': (10.0, 100.0),
    'Q': (0.05, 20.0),
    'e_irr': (1.0, 1.5),
    'R': (0.01, 100.0),
    'r': (0.6, 2.5),
}

fitted_params, eval_metric, _ = fit_nqs(
    h_dicts=h_dicts,
    nn_losses=nn_losses,
    seed=6,
    number_of_initializations=2000,
    param_ranges=param_ranges,
    gtol=1e-8,
    max_steps=5000,
    loss='mse',       # 'huber' for Huber Loss (on log scale) or 'mse' Mean Squared Error (on log scale)
)
```

**Returns:** `(fitted_nqs_dict, eval_metric_value, None)`
- `fitted_nqs_dict`: dict with keys `p, q, P, Q, e_irr, R, r`
- `eval_metric_value`: RMSLE percentage = `sqrt(2 * loss) * 100`

### Note on parameter identifiability

> [!IMPORTANT]
> The NQS fitting objective is non-convex and has approximate degeneracies, so different runs — different seeds, different multistart initializations, or even different LHS implementations — may converge to noticeably different values for `P`, `Q`, `e_irr`, `R`, and `r` while producing nearly identical predictions on held-out data. The exponents `p` and `q` are typically well constrained because they govern the slopes of the loss curves on log-log plots, which the data pins down tightly. The remaining parameters are less identifiable.
>
> In practice, **judge fits by held-out prediction quality, not by parameter-by-parameter agreement across runs.**

## Parameter Reference

**NQS parameters** (`nqs_dict`):

| Key     | Description                                    |
|---------|------------------------------------------------|
| `p`     | Exponent for initial risk per dimension        |
| `q`     | Exponent for eigenvalues                       |
| `P`     | Multiplier for initial risk                    |
| `Q`     | Multiplier for eigenvalues                     |
| `e_irr` | Irreducible error (Bayes risk)                 |
| `R`     | Multiplier for noise variance                  |
| `r`     | Exponent for noise variance                    |

**Training hyperparameters** (`h`):

| Key | Description               |
|-----|---------------------------|
| `N` | Number of model parameters |
| `K` | Number of training steps   |
| `B` | Batch size                 |

## Data Requirements

To obtain a reliable fit, all training runs in the dataset should share the same NN configuration — optimizer, learning rate, learning rate schedule, architecture family, tokenizer, etc. Only the three hyperparameters `N`, `B`, and `K` should vary across runs. This ensures the fitted NQS parameters capture scaling behavior rather than confounding differences in training setup.

For example, the included dataset was produced from a granular (across model sizes) version of the Pythia model family, with model sizes up to 2B and compute budgets between 9e14 and 6e19 FLOPs. The LLMs were trained on OpenWebText2, using a customized BPE tokenizer with a vocabulary size of 3000 and sequence length 128. All runs used an Adam optimizer with a cosine learning rate schedule (1% warmup, initial learning rate 1e-3).

## Dataset Format

The `dataset.csv` used by the demo has columns:

```
N, B, K, C, loss, split, type, filtered
```

- `C`: compute budget (in GFLOPs). Only used for grouping curves in the demo plots (IsoFLOP/IsoToken); not required by any `nqs` package function
- `split`: `train` or `test`
- `type`: `isoflop` (fixed compute, varying N) or `isotoken` (fixed compute, varying B)
- `filtered`: `True` if the row was excluded from fitting. Rows are filtered for two reasons: (1) the observed loss exceeds a threshold (loss > 7.0), indicating a failed or diverged run, or (2) the run falls in a small-batch regime where diminishing returns make the data point uninformative for fitting the scaling model. Filtered rows are retained in the dataset for reference but are not used during parameter fitting.


## Demo

`demo_fit_and_predict.py` runs the full pipeline: load data, fit NQS parameters, predict on train/test sets, and produce evaluation outputs.

```bash
# Fit the model from scratch
python demo_fit_and_predict.py

# Skip fitting, use pre-fitted parameters
python demo_fit_and_predict.py --no-fit
```

**Outputs:**
- `actual_vs_predicted.png` — log-log scatter plot of predicted vs actual loss (NQS vs ground truth)
- `isoflop_isotoken_curves.png` — IsoFLOP and IsoToken scaling curves
- `demo_log.txt` — full log of all printed output
- `fitted_nqs.json` — fitted parameters as JSON (only when fitting is run)

### Demo Results

<img src="actual_vs_predicted.png" width="400">

<img src="isoflop_isotoken_curves.png" width="600">


