"""Generate performance comparison figures: analytical vs MC for y=ax+b."""
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from mcup import WeightedRegressor

ASSETS = "docs/assets"
RNG = np.random.default_rng(42)

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


def line(x, p):
    return p[0] + p[1] * x


def make_data(n, rng):
    x = np.linspace(0, 10, n)
    y_err = 0.5 * np.ones(n)
    y = 1.0 + 2.0 * x + rng.normal(0, y_err)
    return x, y, y_err


REPEATS = 3

# ── Figure 1: time vs number of points ──────────────────────────────────────

n_points = [10, 20, 50, 100, 200, 500]
mc_iters_fixed = 200

times_analytical = []
times_mc = []

print("=== Runtime vs n_points ===", flush=True)
for n in n_points:
    x, y, y_err = make_data(n, RNG)

    t_a = []
    for _ in range(REPEATS):
        est = WeightedRegressor(line, method="analytical")
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(x, y, y_err=y_err, p0=[0.0, 1.0])
        t_a.append(time.perf_counter() - t0)
    times_analytical.append(np.median(t_a))

    t_m = []
    for _ in range(REPEATS):
        est = WeightedRegressor(line, method="mc", n_iter=mc_iters_fixed)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(x, y, y_err=y_err, p0=[0.0, 1.0])
        t_m.append(time.perf_counter() - t0)
    times_mc.append(np.median(t_m))

    print(
        f"n={n:5d}  analytical={times_analytical[-1]*1000:.1f}ms"
        f"  mc={times_mc[-1]*1000:.1f}ms",
        flush=True,
    )

# ── Figure 2: MC convergence vs n_iter (fixed n=50) ─────────────────────────

N_FIXED = 50
x50, y50, ye50 = make_data(N_FIXED, RNG)

# analytical ground truth
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    est_ref = WeightedRegressor(line, method="analytical")
    est_ref.fit(x50, y50, y_err=ye50, p0=[0.0, 1.0])
ref_a, ref_b = est_ref.params_
ref_a_std, ref_b_std = est_ref.params_std_

n_iter_vals = [50, 100, 200, 500, 1000, 2000]
mc_times = []
mc_b_std = []
mc_b_err = []  # |MC_std - analytical_std| / analytical_std

print("\n=== MC convergence vs n_iter (n=50) ===", flush=True)
for n_iter in n_iter_vals:
    runs_std = []
    t_runs = []
    for _ in range(REPEATS):
        est = WeightedRegressor(line, method="mc", n_iter=n_iter)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(x50, y50, y_err=ye50, p0=[0.0, 1.0])
        t_runs.append(time.perf_counter() - t0)
        runs_std.append(est.params_std_[1])
    mc_times.append(np.median(t_runs))
    mc_b_std.append(np.mean(runs_std))
    mc_b_err.append(abs(np.mean(runs_std) - ref_b_std) / ref_b_std * 100)
    print(
        f"n_iter={n_iter:5d}  t={mc_times[-1]*1000:.1f}ms"
        f"  σ_b={mc_b_std[-1]:.5f}  err={mc_b_err[-1]:.1f}%",
        flush=True,
    )

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Panel 1: time vs n_points
ax = axes[0]
ax.plot(
    n_points, [t * 1000 for t in times_analytical], "o-",
    color="#55a868", lw=2, ms=6, label="Analytical",
)
ax.plot(
    n_points, [t * 1000 for t in times_mc], "s--",
    color="#dd8452", lw=2, ms=6, label=f"MC ({mc_iters_fixed} iter)",
)
ax.set_xlabel("Number of data points")
ax.set_ylabel("Time (ms, median of 3 runs)")
ax.set_title("Runtime vs dataset size")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
ax.grid(True, which="both", alpha=0.3)

# Panel 2: time vs n_iter
ax = axes[1]
ax.plot(
    n_iter_vals, [t * 1000 for t in mc_times], "s-",
    color="#dd8452", lw=2, ms=6, label="MC (n=50 points)",
)
ax.axhline(
    times_analytical[n_points.index(50)] * 1000,
    color="#55a868", lw=2, linestyle="--", label="Analytical (n=50)",
)
ax.set_xlabel("MC iterations")
ax.set_ylabel("Time (ms)")
ax.set_title("MC runtime vs iterations (n=50)")
ax.set_xscale("log")
ax.legend()
ax.grid(True, which="both", alpha=0.3)

# Panel 3: σ_b convergence vs n_iter
ax = axes[2]
ax_right = ax.twinx()

ax.plot(n_iter_vals, mc_b_std, "s-", color="#dd8452", lw=2, ms=6, label="MC σ_b")
ax.axhline(
    ref_b_std, color="#55a868", lw=2, linestyle="--",
    label=f"Analytical σ_b = {ref_b_std:.4f}",
)
ax_right.plot(
    n_iter_vals, mc_b_err, "^:", color="#c44e52", lw=1.5, ms=5, alpha=0.7,
    label="Relative error (%)",
)
ax_right.set_ylabel("Relative error vs analytical (%)", color="#c44e52")
ax_right.tick_params(axis="y", labelcolor="#c44e52")

ax.set_xlabel("MC iterations")
ax.set_ylabel("σ_b (slope uncertainty)")
ax.set_title("σ_b convergence vs iterations (n=50)")
ax.set_xscale("log")
ax.grid(True, which="both", alpha=0.3)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_right.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

fig.suptitle(
    "Analytical vs Monte Carlo — runtime and precision (y = a + bx)",
    fontweight="bold",
)
fig.tight_layout()
fig.savefig(f"{ASSETS}/performance_analytical_vs_mc.png", bbox_inches="tight")
plt.close()
print("\nPerformance figure saved.", flush=True)

# Print summary table values for the doc
print("\n--- Table: time vs n_points ---")
print(f"{'n':>6} | {'Analytical (ms)':>16} | {f'MC {mc_iters_fixed} iter (ms)':>17} | {'Speedup':>8}")
print("-" * 58)
for n, ta, tm in zip(n_points, times_analytical, times_mc):
    print(f"{n:>6} | {ta*1000:>16.1f} | {tm*1000:>17.1f} | {tm/ta:>8.1f}x")

print("\n--- Table: MC convergence ---")
print(f"{'n_iter':>8} | {'σ_b':>10} | {'Error vs analytical':>20} | {'Time (ms)':>10}")
print("-" * 58)
for ni, std, err, t in zip(n_iter_vals, mc_b_std, mc_b_err, mc_times):
    print(f"{ni:>8} | {std:>10.5f} | {err:>19.1f}% | {t*1000:>10.1f}")
print(
    f"{'Analytical':>8} | {ref_b_std:>10.5f} | {'0.0':>19}% |"
    f" {times_analytical[n_points.index(50)]*1000:>10.1f}"
)
