"""
Demo: load dataset.csv, fit NQS model on train data, predict on test data,
      and produce evaluation plots and summary table.

Usage:
    python demo_fit_and_predict.py           # fit the model
    python demo_fit_and_predict.py --no-fit  # skip fitting, use pre-fitted params
"""

import csv
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from nqs import fit_nqs, compute_nqs_regularized, compute_nqs_standard

DATASET_CSV = "dataset.csv"
LOG_FILE = "demo_log.txt"
SKIP_FIT = "--no-fit" in sys.argv

# Redirect all print output to both console and log file
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

_log_fh = open(LOG_FILE, 'w')
sys.stdout = Tee(sys.__stdout__, _log_fh)

# ------------------------------------------------------------------ #
#  1. Load dataset                                                    #
# ------------------------------------------------------------------ #

all_rows = []
train_h_dicts = []
train_losses = []
test_rows = []

with open(DATASET_CSV, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        all_rows.append(row)
        if row['split'] == 'train' and row['filtered'] != 'True':
            train_h_dicts.append({
                'N': int(row['N']),
                'K': int(row['K']),
                'B': int(row['B']),
            })
            train_losses.append(float(row['loss']))
        elif row['split'] == 'test':
            test_rows.append(row)

_train_total = sum(1 for r in all_rows if r['split'] == 'train')
_train_filtered = _train_total - len(train_h_dicts)
print(f"Train: {len(train_h_dicts)} runs ({_train_filtered} filtered out of {_train_total})")
print(f"Test:  {len(test_rows)} runs")


# ------------------------------------------------------------------ #
#  2. Fit NQS model (or use pre-fitted params)                        #
# ------------------------------------------------------------------ #

PRE_FITTED_NQS = {
    'p': 1.1131253732848405,
    'q': 0.584242820729791,
    'P': 3.3926855412266743,
    'Q': 0.8853327469240055,
    'e_irr': 0.35449737765209993,
    'R': 8.410028810517371,
    'r': 2.5829307382669966,
}

if SKIP_FIT:
    print("\nSkipping fitting, using pre-fitted NQS parameters")
    fitted_nqs = PRE_FITTED_NQS
    for k, v in fitted_nqs.items():
        print(f"  {k}: {v:.6f}")
else:
    param_ranges = {
        'p': (1.05, 2.5),
        'q': (0.6, 2.5),
        'P': (10.0, 100.0),
        'Q': (0.05, 20.0),
        'e_irr': (1.0, 1.5),
        'R': (0.1**2, 10.0**2),
        'r': (0.6, 2.5),
    }

    fitted_nqs, eval_metric, _ = fit_nqs(
        h_dicts=train_h_dicts,
        nn_losses=train_losses,
        seed=6,
        number_of_initializations=2000,
        param_ranges=param_ranges,
        gtol=1e-8,
        max_steps=5000,
        loss='mse',
    )

    print(f"\nFitted NQS parameters (fitted vs expected):")
    all_close = True
    rtol = 0.02
    for k in PRE_FITTED_NQS:
        fitted_v = fitted_nqs[k]
        expected_v = PRE_FITTED_NQS[k]
        rel_err = abs(fitted_v - expected_v) / abs(expected_v)
        status = "OK" if rel_err < rtol else "FAIL"
        if rel_err >= rtol:
            all_close = False
        print(f"  {k:5s}: fitted={fitted_v:.6f}  expected={expected_v:.6f}  rel_err={rel_err:.4f}  [{status}]")

    print(f"Eval metric (RMSLE %): {eval_metric:.4f}")
    print(f"Parameter check: {'PASS' if all_close else 'FAIL'} (rtol={rtol})")

    with open('fitted_nqs.json', 'w') as jf:
        json.dump(fitted_nqs, jf, indent=2)
    print(f"Fitted params saved to fitted_nqs.json")


# ------------------------------------------------------------------ #
#  3. Predict on train and test data                                  #
# ------------------------------------------------------------------ #

# Initial weight norm squared as a function of N (Pythia default: each weight ~ N(0, 0.02))
init_weight_norm_squared_fn = lambda N: N * 0.02**2

train_nn = []
train_nqs = []
test_nn = []
test_nqs = []

# Train predictions
for h, nn_loss in zip(train_h_dicts, train_losses):
    risk = float(compute_nqs_regularized(fitted_nqs, h, init_weight_norm_squared_fn))
    train_nn.append(nn_loss)
    train_nqs.append(risk)

# Test predictions
print(f"\n{'N':>12s} {'B':>8s} {'K':>8s} {'type':>10s} {'nn_loss':>10s} {'nqs_risk':>10s}")
print("-" * 75)

for row in test_rows:
    h = {
        'N': int(row['N']),
        'K': int(row['K']),
        'B': int(row['B']),
    }

    risk = float(compute_nqs_regularized(fitted_nqs, h, init_weight_norm_squared_fn))
    nn_loss = float(row['loss'])
    test_nn.append(nn_loss)
    test_nqs.append(risk)

    print(f"{h['N']:>12d} {h['B']:>8d} {h['K']:>8d} {row['type']:>10s} {nn_loss:>10.4f} {risk:>10.4f}")


# ------------------------------------------------------------------ #
#  4. Actual vs predicted plot                                        #
# ------------------------------------------------------------------ #

fig, ax = plt.subplots(figsize=(4, 4))

ax.scatter(train_nn, train_nqs, c='blue', label='train', alpha=0.7, s=30)
ax.scatter(test_nn, test_nqs, c='red', label='test', alpha=0.7, s=30)

# Diagonal reference line
all_vals = train_nn + test_nn + train_nqs + test_nqs
lo, hi = min(all_vals), max(all_vals)
ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.8)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('LLM loss (actual)')
ax.set_ylabel('NQS risk (predicted)')
ax.set_title('Actual vs Predicted', fontweight='bold')
ax.legend()
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150)
plt.close()
print(f"\nPlot saved to actual_vs_predicted.png")


# ------------------------------------------------------------------ #
#  5. Evaluation metrics (Huber + MSE in log space)                   #
# ------------------------------------------------------------------ #

def huber_loss_log(y_true, y_pred, delta=1e-3):
    """Huber loss on log scale for arrays."""
    diff = np.log(y_true) - np.log(y_pred)
    abs_diff = np.abs(diff)
    return np.where(abs_diff <= delta, 0.5 * diff**2, delta * (abs_diff - 0.5 * delta))

def avg_huber(nn, nqs, delta=1e-3):
    nn, nqs = np.array(nn), np.array(nqs)
    if len(nn) == 0:
        return float('nan')
    return float(np.mean(huber_loss_log(nn, nqs, delta)))

def avg_mse(nn, nqs):
    nn, nqs = np.array(nn), np.array(nqs)
    if len(nn) == 0:
        return float('nan')
    return float(np.mean((np.log(nn) - np.log(nqs))**2))

# Compute NQS predictions for all rows (train unfiltered + test)
results = []  # list of dicts with nn_loss, nqs_risk, split, type, N, B, K, C

for row in all_rows:
    split = row['split']
    filtered = row.get('filtered', '') == 'True'

    # Skip filtered train rows
    if split == 'train' and filtered:
        continue

    h = {
        'N': int(row['N']),
        'K': int(row['K']),
        'B': int(row['B']),
    }
    risk = float(compute_nqs_regularized(fitted_nqs, h, init_weight_norm_squared_fn))
    results.append({
        'nn_loss': float(row['loss']),
        'nqs_risk': risk,
        'split': split,
        'type': row['type'],
        'N': h['N'],
        'B': h['B'],
        'K': h['K'],
        'C': int(row['C']),
    })

# Split results for metrics
train_res = [r for r in results if r['split'] == 'train']
test_isoflop = [r for r in results if r['split'] == 'test' and r['type'] == 'isoflop']
test_isotoken = [r for r in results if r['split'] == 'test' and r['type'] == 'isotoken']

def extract(lst):
    return [r['nn_loss'] for r in lst], [r['nqs_risk'] for r in lst]

print(f"\n{'='*60}")
print("Average Huber Loss (log scale, delta=1e-3)")
print(f"{'='*60}")
print(f"  {'Split':<20s} {'N':>5s}  {'Avg Huber':>12s}  {'Avg MSE':>12s}")
print(f"  {'-'*55}")

for label, subset in [('Train', train_res), ('Test IsoFLOP', test_isoflop), ('Test IsoToken', test_isotoken)]:
    nn, nqs = extract(subset)
    h_val = avg_huber(nn, nqs)
    m_val = avg_mse(nn, nqs)
    print(f"  {label:<20s} {len(subset):>5d}  {h_val:>12.6f}  {m_val:>12.6f}")

print(f"{'='*60}")


# ------------------------------------------------------------------ #
#  6. IsoFLOP and IsoToken curves                                     #
# ------------------------------------------------------------------ #

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(6.5, 3.0))

color_map = {'train': 'blue', 'test': 'red'}

# Only include compute budgets near these target values (in PF), with 10% tolerance
TARGET_PF = [0.92, 4, 15, 59, 236, 944, 3780, 15100, 60500]
TARGET_C = [pf * 1e6 for pf in TARGET_PF]  # convert PF to GFLOPs (C units)

def is_target_C(c_val):
    return any(abs(c_val - t) / t < 0.10 for t in TARGET_C)

def format_C(c):
    c_pf = c / 1e6
    if c_pf >= 1:
        return f"  {int(round(c_pf))} PF"
    return f"  {c_pf:.2f} PF"

# --- IsoFLOP: loss vs N for each C ---
isoflop_res = [r for r in results if r['type'] == 'isoflop']
isoflop_Cs = sorted(c for c in set(r['C'] for r in isoflop_res) if is_target_C(c))
isoflop_label_Cs = set()
for split in ['train', 'test']:
    split_Cs = [c for c in isoflop_Cs if any(r['C'] == c and r['split'] == split for r in isoflop_res)]
    if split_Cs:
        isoflop_label_Cs.add(max(split_Cs))

for c_val in isoflop_Cs:
    group = sorted([r for r in isoflop_res if r['C'] == c_val], key=lambda r: r['N'])
    Ns = [r['N'] for r in group]
    nn = [r['nn_loss'] for r in group]
    nqs = [r['nqs_risk'] for r in group]
    color = color_map[group[0]['split']]

    ax_left.scatter(Ns, nn, color=color, marker='o', s=40, alpha=0.7, ec='none')
    ax_left.plot(Ns, nqs, color=color, linestyle='-', linewidth=2, alpha=0.7)

    if c_val in isoflop_label_Cs:
        label = format_C(c_val)
        ax_left.text(Ns[-1] * 1.05, nqs[-1], label, color=color, fontsize=8, fontweight='bold', va='center')

ax_left.set_xscale("log")
ax_left.set_yscale("log")
ax_left.set_xlabel("Parameters Count")
ax_left.set_ylabel("Loss")
ax_left.set_title("IsoFLOP Curves", fontweight='bold')
x_min, x_max = ax_left.get_xlim()
ax_left.set_xlim(x_min * 0.5, x_max * 2.5)

# --- IsoToken: loss vs B for each C ---
isotoken_res = [r for r in results if r['type'] == 'isotoken']
isotoken_Cs = sorted(c for c in set(r['C'] for r in isotoken_res) if is_target_C(c))
isotoken_label_Cs = set()
for split in ['train', 'test']:
    split_Cs = [c for c in isotoken_Cs if any(r['C'] == c and r['split'] == split for r in isotoken_res)]
    if split_Cs:
        isotoken_label_Cs.add(max(split_Cs))

for c_val in isotoken_Cs:
    group = sorted([r for r in isotoken_res if r['C'] == c_val], key=lambda r: r['B'])
    Bs = [r['B'] for r in group]
    nn = [r['nn_loss'] for r in group]
    nqs = [r['nqs_risk'] for r in group]
    color = color_map[group[0]['split']]

    ax_right.scatter(Bs, nn, color=color, marker='o', s=40, alpha=0.7, ec='none')
    ax_right.plot(Bs, nqs, color=color, linestyle='-', linewidth=2, alpha=0.7)

    if c_val in isotoken_label_Cs:
        label = format_C(c_val)
        ax_right.text(Bs[-1] * 1.05, nqs[-1], label, color=color, fontsize=8, fontweight='bold', va='center')

ax_right.set_xscale("log")
ax_right.set_yscale("log")
ax_right.set_xlabel("Batch Size")
ax_right.set_ylabel("Loss")
ax_right.set_title("IsoToken Curves", fontweight='bold')
x_min, x_max = ax_right.get_xlim()
ax_right.set_xlim(x_min * 0.5, x_max * 2.5)

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='gray', linestyle='None', markersize=8, label='LLM (actual)'),
    Line2D([0], [0], color='gray', linestyle='-', linewidth=2, label='NQS (predicted)'),
    Line2D([0], [0], color='blue', linestyle='-', linewidth=4, label='Train'),
    Line2D([0], [0], color='red', linestyle='-', linewidth=4, label='Test'),
]
fig.legend(handles=legend_elements, fontsize=9, loc='lower center',
           bbox_to_anchor=(0.5, -0.08), ncol=4)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig('isoflop_isotoken_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot saved to isoflop_isotoken_curves.png")

print(f"\nLog saved to {LOG_FILE}")
_log_fh.close()
sys.stdout = sys.__stdout__

