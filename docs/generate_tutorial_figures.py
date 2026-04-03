"""Generate tutorial figures for the MCUP docs.

Error magnitudes are chosen so the failure mode of the naive method is
immediately visible: biased fits, wildly over-confident intervals, confidence
bands that are 5–20× too wide, or coverage that deviates badly from nominal.
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from mcup import WeightedRegressor, XYWeightedRegressor, DemingRegressor

ASSETS = "docs/assets"
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


def linear_band(x_plot, cov):
    """1σ confidence band on a + b*x given 2×2 covariance [[σ²a,cab],[cab,σ²b]]."""
    sa2, cab, sb2 = cov[0, 0], cov[0, 1], cov[1, 1]
    return np.sqrt(sa2 + 2 * x_plot * cab + x_plot**2 * sb2)


# ── Tutorial 1: Sensor Calibration ──────────────────────────────────────────
# Extreme heteroscedastic noise: low-V points are almost noiseless (σ≈0.05 V),
# high-V points are very noisy (σ≈0.45 V — 9:1 ratio).
# OLS confidence band is 5× wider than WLS — made visible by shading.

np.random.seed(42)
a_true, b_true = 0.05, 1.02
V_ref = np.linspace(0.5, 5.0, 30)
sigma = 0.01 + 0.08 * V_ref          # 0.05 V … 0.41 V  (9:1 ratio)
V_sensor = a_true + b_true * V_ref + np.random.normal(0, sigma)

def calibration_line(x, p):
    return p[0] + p[1] * x

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    est_ols = WeightedRegressor(calibration_line, method="analytical")
    est_ols.fit(V_ref, V_sensor, y_err=np.ones_like(V_ref), p0=[0.0, 1.0])

    est_w = WeightedRegressor(calibration_line, method="analytical")
    est_w.fit(V_ref, V_sensor, y_err=sigma, p0=[0.0, 1.0])

x_plot = np.linspace(0.3, 5.5, 200)
ols_fit = est_ols.params_[0] + est_ols.params_[1] * x_plot
wls_fit = est_w.params_[0] + est_w.params_[1] * x_plot
ols_err = linear_band(x_plot, est_ols.covariance_)
wls_err = linear_band(x_plot, est_w.covariance_)

ratio = est_ols.params_std_[1] / est_w.params_std_[1]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.errorbar(V_ref, V_sensor, yerr=sigma, fmt="o", color="#4c72b0",
            capsize=3, ms=5, label="Data (±σ_y)", zorder=3)
# OLS: show fit + shaded 1σ confidence band
ax.plot(x_plot, ols_fit, "--", color="#dd8452", lw=2.0,
        label=f"OLS  b={est_ols.params_[1]:.3f} ± {est_ols.params_std_[1]:.3f}")
ax.fill_between(x_plot, ols_fit - ols_err, ols_fit + ols_err,
                color="#dd8452", alpha=0.20, label="OLS 1σ band")
# WLS: fit + narrow confidence band
ax.plot(x_plot, wls_fit, "-", color="#55a868", lw=2.2,
        label=f"Weighted  b={est_w.params_[1]:.3f} ± {est_w.params_std_[1]:.4f}")
ax.fill_between(x_plot, wls_fit - wls_err, wls_fit + wls_err,
                color="#55a868", alpha=0.35, label="WLS 1σ band")
ax.plot(x_plot, a_true + b_true * x_plot, ":", color="#333", lw=1.5,
        label=f"True  b={b_true}")
ax.set_xlabel("Reference voltage (V)")
ax.set_ylabel("Sensor voltage (V)")
ax.set_title("OLS confidence band is far wider than WLS")
ax.legend(fontsize=8.5)

ax = axes[1]
params = ["Offset a (V)", "Gain b"]
x_pos = np.arange(2)
width = 0.25
ax.bar(x_pos - width, [a_true, b_true], width, color="#aaa", label="True value")
ax.bar(x_pos,        est_ols.params_, width, yerr=est_ols.params_std_, capsize=6,
       color="#dd8452", alpha=0.85, label="OLS (uniform σ=1)")
ax.bar(x_pos + width, est_w.params_,  width, yerr=est_w.params_std_, capsize=6,
       color="#55a868", alpha=0.85, label="WeightedRegressor")
ax.set_xticks(x_pos)
ax.set_xticklabels(params)
ax.set_title(f"OLS σ_b is {ratio:.0f}× too wide — wasted precision")
ax.legend(fontsize=9)

fig.suptitle(
    "Tutorial 1 — Sensor Calibration: heteroscedastic noise (σ = 0.01 + 0.08·V)",
    fontweight="bold")
fig.tight_layout()
fig.savefig(f"{ASSETS}/tutorial1_sensor_calibration.png", bbox_inches="tight")
plt.close()
print(f"Tutorial 1: OLS b={est_ols.params_[1]:.3f}±{est_ols.params_std_[1]:.4f}  "
      f"WLS b={est_w.params_[1]:.3f}±{est_w.params_std_[1]:.4f}  "
      f"ratio={ratio:.1f}x")
print("Tutorial 1 figure saved.")


# ── Tutorial 2: Radioactive Decay ───────────────────────────────────────────
# σ_t = 15 s — far larger than the 2 s used previously.
# At early time points (t = 5 s), ±15 s represents 300 % of the time value.
# Effect: y-only fit is BIASED (λ overestimated ≈ +30 %) AND has a 3× too-small
# σ_λ.  XYWeightedRegressor recovers the true λ with honest error bars.

np.random.seed(7)
A0_true = 1000.0
lambda_true = 0.05
t_true = np.array([0, 5, 10, 20, 35, 50, 70, 90, 120, 150], dtype=float)
A_true = A0_true * np.exp(-lambda_true * t_true)
sigma_A = np.sqrt(A_true)
A_meas = np.clip(A_true + np.random.normal(0, sigma_A), 1.0, None)
sigma_t = 15.0 * np.ones_like(t_true)           # was 2 s
t_meas = np.maximum(t_true + np.random.normal(0, sigma_t), 0.0)

def decay(t, p):
    return p[0] * np.exp(-p[1] * t)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    est_y = WeightedRegressor(decay, method="analytical")
    est_y.fit(t_meas, A_meas, y_err=sigma_A, p0=[900.0, 0.04])

    est_xy = XYWeightedRegressor(decay, method="analytical")
    est_xy.fit(t_meas, A_meas, x_err=sigma_t, y_err=sigma_A, p0=[900.0, 0.04])

bias_pct = (est_y.params_[1] - lambda_true) / lambda_true * 100
ratio2 = est_xy.params_std_[1] / est_y.params_std_[1]
t_plot = np.linspace(0, 160, 300)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.errorbar(t_meas, A_meas, xerr=sigma_t, yerr=sigma_A, fmt="o",
            color="#4c72b0", capsize=3, ms=5, label="Data (±σ_t=15 s, ±σ_A)", zorder=3)
ax.plot(t_plot, decay(t_plot, [A0_true, lambda_true]), ":",
        color="#333", lw=1.5, label=f"True  λ={lambda_true}")
ax.plot(t_plot, decay(t_plot, est_y.params_), "--",
        color="#dd8452", lw=2.0,
        label=f"y-only  λ={est_y.params_[1]:.4f}±{est_y.params_std_[1]:.4f}")
ax.plot(t_plot, decay(t_plot, est_xy.params_), "-",
        color="#55a868", lw=2.2,
        label=f"XY  λ={est_xy.params_[1]:.4f}±{est_xy.params_std_[1]:.4f}")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Activity (Bq)")
ax.set_title(f"y-only fit biased by {bias_pct:+.0f}%  — XY recovers truth")
ax.legend(fontsize=9)

ax = axes[1]
labels = ["y-errors only\n(ignoring σ_t)", "XY errors\n(XYWeightedRegressor)"]
lam_vals = [est_y.params_[1], est_xy.params_[1]]
lam_stds = [est_y.params_std_[1], est_xy.params_std_[1]]
bars = ax.bar(labels, lam_vals, yerr=lam_stds, capsize=10,
              color=["#dd8452", "#55a868"], alpha=0.85, width=0.4)
ax.axhline(lambda_true, color="#333", linestyle=":", lw=2,
           label=f"True λ = {lambda_true}")
ax.set_ylabel("Decay constant λ (s⁻¹)")
ax.set_title(f"σ_λ underestimated {ratio2:.1f}×  +  biased point estimate")
ax.legend(fontsize=9)

# annotate bias gap
ax.annotate(f"Bias\n{bias_pct:+.0f}%",
            xy=(0, lambda_true), xytext=(-0.3, (lam_vals[0] + lambda_true) / 2),
            fontsize=9, ha="center", color="#dd8452",
            arrowprops=dict(arrowstyle="-|>", color="#dd8452",
                            connectionstyle="arc3,rad=0.2"))

fig.suptitle("Tutorial 2 — Radioactive Decay: σ_t = 15 s (large timing errors)",
             fontweight="bold")
fig.tight_layout()
fig.savefig(f"{ASSETS}/tutorial2_radioactive_decay.png", bbox_inches="tight")
plt.close()
print(f"Tutorial 2: y-only λ={est_y.params_[1]:.4f}±{est_y.params_std_[1]:.4f}  "
      f"XY λ={est_xy.params_[1]:.4f}±{est_xy.params_std_[1]:.4f}  "
      f"ratio={ratio2:.2f}x  bias={bias_pct:+.1f}%")
print("Tutorial 2 figure saved.")


# ── Tutorial 3: Method Comparison (Deming) ──────────────────────────────────
# σ_x = 25 µg/mL on a range 10–200 µg/mL causes severe regression dilution.
# Expected: OLS slope ≈ 0.85 (attenuated 19 % below truth), Deming ≈ 1.05.
# Visual message: "OLS says the methods disagree — Deming says they agree."

np.random.seed(7)
c_true = np.random.uniform(10, 200, 60)
a_true_mc, b_true_mc = 3.0, 1.05
sigma_x, sigma_y = 25.0, 8.0
x_meas = c_true + np.random.normal(0, sigma_x, 60)
y_meas = a_true_mc + b_true_mc * c_true + np.random.normal(0, sigma_y, 60)
x_err_arr = sigma_x * np.ones(60)
y_err_arr = sigma_y * np.ones(60)

def line(x, p):
    return p[0] + p[1] * x

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    est_ols = WeightedRegressor(line, method="analytical")
    est_ols.fit(x_meas, y_meas, y_err=y_err_arr, p0=[0.0, 1.0])

    est_dem = DemingRegressor(line, method="analytical")
    est_dem.fit(x_meas, y_meas, x_err=x_err_arr, y_err=y_err_arr, p0=[0.0, 1.0])

x_plot = np.linspace(-10, 220, 300)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.errorbar(x_meas, y_meas, xerr=x_err_arr, yerr=y_err_arr, fmt="o",
            color="#4c72b0", capsize=2, ms=4, alpha=0.65,
            label=f"Samples (±σ_x={sigma_x}, ±σ_y={sigma_y})", zorder=3)
ax.plot(x_plot, x_plot, ":", color="#444", lw=1.5, label="Identity (y = x)")
ax.plot(x_plot, line(x_plot, est_ols.params_), "--",
        color="#dd8452", lw=2.0,
        label=f"OLS  b={est_ols.params_[1]:.3f}±{est_ols.params_std_[1]:.3f}  ← attenuated")
ax.plot(x_plot, line(x_plot, est_dem.params_), "-",
        color="#55a868", lw=2.2,
        label=f"Deming  b={est_dem.params_[1]:.3f}±{est_dem.params_std_[1]:.3f}  ← correct")
ax.set_xlim(-10, 230)
ax.set_xlabel("Old method (μg/mL)")
ax.set_ylabel("New method (μg/mL)")
ax.set_title(f"OLS slope {est_ols.params_[1]:.2f} vs Deming {est_dem.params_[1]:.2f}  (true {b_true_mc})")
ax.legend(fontsize=9)

ax = axes[1]
params_labels = ["Intercept a", "Slope b"]
x_pos = np.arange(2)
width = 0.25
ax.bar(x_pos - width, [a_true_mc, b_true_mc], width, color="#aaa", label="True value")
ax.bar(x_pos,         est_ols.params_, width, yerr=est_ols.params_std_, capsize=6,
       color="#dd8452", alpha=0.85, label="OLS (ignores x-err)")
ax.bar(x_pos + width, est_dem.params_, width, yerr=est_dem.params_std_, capsize=6,
       color="#55a868", alpha=0.85, label="DemingRegressor")
ax.set_xticks(x_pos)
ax.set_xticklabels(params_labels)
ols_bias = (est_ols.params_[1] - b_true_mc) / b_true_mc * 100
ax.set_title(f"OLS slope biased {ols_bias:+.0f}% — Deming recovers truth")
ax.legend(fontsize=9)

fig.suptitle(
    f"Tutorial 3 — Method Comparison: σ_x={sigma_x} causes regression dilution",
    fontweight="bold")
fig.tight_layout()
fig.savefig(f"{ASSETS}/tutorial3_method_comparison.png", bbox_inches="tight")
plt.close()
print(f"Tutorial 3: OLS b={est_ols.params_[1]:.3f}±{est_ols.params_std_[1]:.3f}  "
      f"Deming b={est_dem.params_[1]:.3f}±{est_dem.params_std_[1]:.3f}  "
      f"true={b_true_mc}")
print("Tutorial 3 figure saved.")
print("All tutorial figures done.")
