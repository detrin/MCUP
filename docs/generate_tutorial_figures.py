"""Generate tutorial figures for the MCUP docs."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mcup import WeightedRegressor, XYWeightedRegressor, DemingRegressor

ASSETS = "docs/assets"
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


# ── Tutorial 1: Sensor Calibration ──────────────────────────────────────────

np.random.seed(42)
a_true, b_true = 0.05, 1.02
V_ref = np.linspace(0.5, 5.0, 20)
sigma = 0.01 + 0.02 * V_ref
V_sensor = a_true + b_true * V_ref + np.random.normal(0, sigma)

def calibration_line(x, p):
    return p[0] + p[1] * x

# OLS (uniform weights = 1)
est_ols = WeightedRegressor(calibration_line, method="analytical")
est_ols.fit(V_ref, V_sensor, y_err=np.ones_like(V_ref), p0=[0.0, 1.0])

# Weighted
est_w = WeightedRegressor(calibration_line, method="analytical")
est_w.fit(V_ref, V_sensor, y_err=sigma, p0=[0.0, 1.0])

x_plot = np.linspace(0.3, 5.2, 200)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.errorbar(V_ref, V_sensor, yerr=sigma, fmt="o", color="#4c72b0",
            capsize=3, ms=5, label="Data (±σ_y)", zorder=3)
ax.plot(x_plot, calibration_line(x_plot, est_ols.params_), "--",
        color="#dd8452", lw=1.8, label=f"OLS  b={est_ols.params_[1]:.4f}±{est_ols.params_std_[1]:.4f}")
ax.plot(x_plot, calibration_line(x_plot, est_w.params_), "-",
        color="#55a868", lw=2, label=f"Weighted  b={est_w.params_[1]:.4f}±{est_w.params_std_[1]:.4f}")
ax.plot(x_plot, a_true + b_true * x_plot, ":",
        color="#666", lw=1.5, label=f"True  b={b_true}")
ax.set_xlabel("Reference voltage (V)")
ax.set_ylabel("Sensor voltage (V)")
ax.set_title("Fit comparison")
ax.legend(fontsize=9)

ax = axes[1]
params = ["Offset a (V)", "Gain b"]
true_vals = [a_true, b_true]
ols_vals = est_ols.params_
ols_stds = est_ols.params_std_
w_vals = est_w.params_
w_stds = est_w.params_std_

x_pos = np.arange(2)
width = 0.25
ax.bar(x_pos - width, true_vals, width, color="#aaa", label="True value")
ax.bar(x_pos, ols_vals, width, yerr=ols_stds, capsize=5,
       color="#dd8452", alpha=0.85, label="OLS")
ax.bar(x_pos + width, w_vals, width, yerr=w_stds, capsize=5,
       color="#55a868", alpha=0.85, label="Weighted")
ax.set_xticks(x_pos)
ax.set_xticklabels(params)
ax.set_title("Parameter estimates ± 1σ")
ax.legend(fontsize=9)

fig.suptitle("Tutorial 1 — Sensor Calibration", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{ASSETS}/tutorial1_sensor_calibration.png", bbox_inches="tight")
plt.close()
print("Tutorial 1 figure saved.")


# ── Tutorial 2: Radioactive Decay ───────────────────────────────────────────

np.random.seed(42)
A0_true = 1000.0
lambda_true = 0.05
t_true = np.array([0, 5, 10, 20, 35, 50, 70, 90, 120, 150], dtype=float)
A_true = A0_true * np.exp(-lambda_true * t_true)
sigma_A = np.sqrt(A_true)
A_meas = np.clip(A_true + np.random.normal(0, sigma_A), 1.0, None)
sigma_t = 2.0 * np.ones_like(t_true)
t_meas = t_true + np.random.normal(0, sigma_t)

def decay(t, p):
    A0, lam = p
    return A0 * np.exp(-lam * t)

est_xy = XYWeightedRegressor(decay, method="analytical")
est_xy.fit(t_meas, A_meas, x_err=sigma_t, y_err=sigma_A, p0=[900.0, 0.04])

est_y = WeightedRegressor(decay, method="analytical")
est_y.fit(t_meas, A_meas, y_err=sigma_A, p0=[900.0, 0.04])

t_plot = np.linspace(0, 160, 300)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.errorbar(t_meas, A_meas, xerr=sigma_t, yerr=sigma_A, fmt="o",
            color="#4c72b0", capsize=3, ms=5, label="Data (±σ_t, ±σ_A)", zorder=3)
ax.plot(t_plot, decay(t_plot, [A0_true, lambda_true]), ":",
        color="#666", lw=1.5, label="True decay")
ax.plot(t_plot, decay(t_plot, est_y.params_), "--",
        color="#dd8452", lw=1.8,
        label=f"y-errors only  λ={est_y.params_[1]:.4f}±{est_y.params_std_[1]:.4f}")
ax.plot(t_plot, decay(t_plot, est_xy.params_), "-",
        color="#55a868", lw=2,
        label=f"XY errors  λ={est_xy.params_[1]:.4f}±{est_xy.params_std_[1]:.4f}")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Activity (Bq)")
ax.set_title("Radioactive decay fit")
ax.legend(fontsize=9)

ax = axes[1]
labels = ["Ignoring t-errors\n(WeightedRegressor)", "Propagating t-errors\n(XYWeightedRegressor)"]
lam_vals = [est_y.params_[1], est_xy.params_[1]]
lam_stds = [est_y.params_std_[1], est_xy.params_std_[1]]
colors = ["#dd8452", "#55a868"]

bars = ax.bar(labels, lam_vals, yerr=lam_stds, capsize=8,
              color=colors, alpha=0.85, width=0.4)
ax.axhline(lambda_true, color="#333", linestyle=":", lw=1.5, label=f"True λ = {lambda_true}")
ax.set_ylabel("Decay constant λ (s⁻¹)")
ax.set_title("λ estimate ± 1σ")
ax.legend(fontsize=9)

# annotate underestimate
ratio = est_xy.params_std_[1] / est_y.params_std_[1]
ax.annotate(f"σ ratio: {ratio:.2f}×",
            xy=(1, lam_vals[1] + lam_stds[1]),
            xytext=(0.5, lam_vals[1] + lam_stds[1] * 1.8),
            fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="->", color="#333"))

fig.suptitle("Tutorial 2 — Radioactive Decay", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{ASSETS}/tutorial2_radioactive_decay.png", bbox_inches="tight")
plt.close()
print("Tutorial 2 figure saved.")


# ── Tutorial 3: Method Comparison ───────────────────────────────────────────

np.random.seed(42)
c_true = np.random.uniform(10, 200, 25)
a_true_mc, b_true_mc = 3.0, 1.05
sigma_x, sigma_y = 5.0, 3.0
x_meas = c_true + np.random.normal(0, sigma_x, 25)
y_meas = a_true_mc + b_true_mc * c_true + np.random.normal(0, sigma_y, 25)
x_err = sigma_x * np.ones(25)
y_err_arr = sigma_y * np.ones(25)

def line(x, p):
    return p[0] + p[1] * x

est_ols = WeightedRegressor(line, method="analytical")
est_ols.fit(x_meas, y_meas, y_err=y_err_arr, p0=[0.0, 1.0])

est_dem = DemingRegressor(line, method="analytical")
est_dem.fit(x_meas, y_meas, x_err=x_err, y_err=y_err_arr, p0=[0.0, 1.0])

x_plot = np.linspace(0, 220, 300)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.errorbar(x_meas, y_meas, xerr=x_err, yerr=y_err_arr, fmt="o",
            color="#4c72b0", capsize=3, ms=5, alpha=0.7,
            label="Samples (±σ_x, ±σ_y)", zorder=3)
ax.plot(x_plot, x_plot, ":", color="#666", lw=1.5, label="Identity (y = x)")
ax.plot(x_plot, line(x_plot, est_ols.params_), "--",
        color="#dd8452", lw=1.8,
        label=f"OLS  b={est_ols.params_[1]:.3f}±{est_ols.params_std_[1]:.3f}")
ax.plot(x_plot, line(x_plot, est_dem.params_), "-",
        color="#55a868", lw=2,
        label=f"Deming  b={est_dem.params_[1]:.3f}±{est_dem.params_std_[1]:.3f}")
ax.set_xlabel("Old method (μg/mL)")
ax.set_ylabel("New method (μg/mL)")
ax.set_title("Method comparison")
ax.legend(fontsize=9)

ax = axes[1]
params = ["Intercept a (μg/mL)", "Slope b"]
true_vals_mc = [a_true_mc, b_true_mc]
ols_v = est_ols.params_
ols_s = est_ols.params_std_
dem_v = est_dem.params_
dem_s = est_dem.params_std_

x_pos = np.arange(2)
width = 0.25
ax.bar(x_pos - width, true_vals_mc, width, color="#aaa", label="True value")
ax.bar(x_pos, ols_v, width, yerr=ols_s, capsize=5,
       color="#dd8452", alpha=0.85, label="OLS (biased)")
ax.bar(x_pos + width, dem_v, width, yerr=dem_s, capsize=5,
       color="#55a868", alpha=0.85, label="Deming")
ax.set_xticks(x_pos)
ax.set_xticklabels(params)
ax.set_title("Parameter estimates ± 1σ")
ax.legend(fontsize=9)

fig.suptitle("Tutorial 3 — Method Comparison (Deming Regression)", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{ASSETS}/tutorial3_method_comparison.png", bbox_inches="tight")
plt.close()
print("Tutorial 3 figure saved.")
print("All done.")
