"""Generate figure for Tutorial 4: multivariable regression."""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from mcup import WeightedRegressor, XYWeightedRegressor

ASSETS = "docs/assets"
RNG = np.random.default_rng(42)

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

# ── Scenario: heat output = a*current^2 + b*temperature + c
# current is measured (error 0.05 A), temperature is controlled (error 0)
# ─────────────────────────────────────────────────────────────────────────────

TRUE_PARAMS = np.array([1.5, 0.3, -2.0])  # a, b, c
N = 60

def heat_model(x, p):
    """Power = p[0]*I^2 + p[1]*T + p[2]  (x = [I, T])"""
    I, T = x[0], x[1]
    return p[0] * I**2 + p[1] * T + p[2]

current = RNG.uniform(0.5, 4.0, N)
temperature = np.linspace(10.0, 40.0, N)
X_true = np.column_stack([current, temperature])

I_err = 0.05 * np.ones(N)
y_err = 0.5 * np.ones(N)
x_err = np.column_stack([I_err, np.zeros(N)])

y_true = np.array([heat_model(X_true[i], TRUE_PARAMS) for i in range(N)])
X_obs = X_true + RNG.normal(0, 1, X_true.shape) * x_err
y_obs = y_true + RNG.normal(0, y_err)

p0 = [1.0, 0.5, 0.0]

# Fit with y-only errors (ignores current error)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    est_y = WeightedRegressor(heat_model, method="analytical")
    est_y.fit(X_obs, y_obs, y_err=y_err, p0=p0)

# Fit with xy errors (accounts for current error)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    est_xy = XYWeightedRegressor(heat_model, method="analytical")
    est_xy.fit(X_obs, y_obs, x_err=x_err, y_err=y_err, p0=p0)

print("True params:        a={:.3f}  b={:.3f}  c={:.3f}".format(*TRUE_PARAMS))
print("WeightedRegressor:  a={:.3f}±{:.3f}  b={:.3f}±{:.3f}  c={:.3f}±{:.3f}".format(
    est_y.params_[0], est_y.params_std_[0],
    est_y.params_[1], est_y.params_std_[1],
    est_y.params_[2], est_y.params_std_[2],
))
print("XYWeightedRegressor:a={:.3f}±{:.3f}  b={:.3f}±{:.3f}  c={:.3f}±{:.3f}".format(
    est_xy.params_[0], est_xy.params_std_[0],
    est_xy.params_[1], est_xy.params_std_[1],
    est_xy.params_[2], est_xy.params_std_[2],
))

# ── Figure: predicted vs observed, residuals ─────────────────────────────────

y_pred_y = np.array([heat_model(X_obs[i], est_y.params_) for i in range(N)])
y_pred_xy = np.array([heat_model(X_obs[i], est_xy.params_) for i in range(N)])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel 1: predicted vs observed
ax = axes[0]
ax.scatter(y_obs, y_true, s=18, alpha=0.5, color="gray", label="Data", zorder=2)
mn, mx = y_obs.min(), y_obs.max()
ax.plot([mn, mx], [mn, mx], "k--", lw=1, label="y=x (perfect)")
ax.set_xlabel("Observed power (W)")
ax.set_ylabel("True power (W)")
ax.set_title("Observations vs truth")
ax.legend(fontsize=9)

# Panel 2: parameter comparison
methods = ["True", "W (y-only)", "XYW (x+y)"]
a_vals = [TRUE_PARAMS[0], est_y.params_[0], est_xy.params_[0]]
a_errs = [0, est_y.params_std_[0], est_xy.params_std_[0]]
b_vals = [TRUE_PARAMS[1], est_y.params_[1], est_xy.params_[1]]
b_errs = [0, est_y.params_std_[1], est_xy.params_std_[1]]

x_pos = np.arange(len(methods))
width = 0.35
colors_a = ["#555555", "#dd8452", "#55a868"]
colors_b = ["#888888", "#dd8452", "#55a868"]

ax2 = axes[1]
bars_a = ax2.bar(x_pos - width / 2, a_vals, width, yerr=a_errs, capsize=4,
                  color=colors_a, alpha=0.85, label="a (quadratic coeff)")
bars_b = ax2.bar(x_pos + width / 2, b_vals, width, yerr=b_errs, capsize=4,
                  color=colors_b, alpha=0.55, label="b (temperature coeff)")
ax2.axhline(0, color="k", lw=0.5)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(methods)
ax2.set_ylabel("Parameter value")
ax2.set_title("Parameter estimates (error bars = 1σ)")
ax2.legend(fontsize=9)

fig.suptitle("Tutorial 4: multivariable regression  P = a·I² + b·T + c",
             fontweight="bold")
fig.tight_layout()
fig.savefig(f"{ASSETS}/tutorial4_multivariable.png", bbox_inches="tight")
plt.close()
print(f"\nFigure saved to {ASSETS}/tutorial4_multivariable.png")
