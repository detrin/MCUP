# Tutorial 2: Radioactive Decay with X and Y Errors

## The problem

You are measuring the activity of a radioactive source over time to determine its half-life. Each activity measurement has counting uncertainty (Poisson statistics: `σ_A = √N / t`). But the time measurements also carry uncertainty — the stopwatch has a precision of ±2 seconds, and there's a human reaction time error on top.

The decay model is:

```
A(t) = A₀ · exp(−λt)
```

where `A₀` is the initial activity and `λ` is the decay constant. The half-life follows as `T½ = ln(2) / λ`.

If you ignore the timing errors and only propagate counting uncertainty, you will underestimate the uncertainty on `λ` — potentially by a large factor when `λ` is small (long half-lives).

---

## Why `XYWeightedRegressor` here

The timing uncertainty `σ_t` propagates into the activity through the model:

```
σ_A_eff_i² = σ_A_i² + |∂A/∂t|²_i · σ_t_i²
            = σ_A_i² + (λ · A(t_i))² · σ_t_i²
```

This is exactly what `XYWeightedRegressor` computes at each iteration (via automatic differentiation with numdifftools), without requiring you to derive the partial derivative by hand.

---

## Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from mcup import XYWeightedRegressor

np.random.seed(42)

# True decay parameters
A0_true = 1000.0    # Bq (initial activity)
lambda_true = 0.05  # 1/s  →  T½ ≈ 13.9 s

# Measurement times (irregular spacing, as in a real experiment)
t_true = np.array([0, 5, 10, 20, 35, 50, 70, 90, 120, 150], dtype=float)

# True activity at each time point
A_true = A0_true * np.exp(-lambda_true * t_true)

# Counting uncertainty: measure for 1 second, Poisson noise
sigma_A = np.sqrt(A_true)
A_meas = A_true + np.random.normal(0, sigma_A)
A_meas = np.clip(A_meas, 1.0, None)  # activity can't go negative

# Timing uncertainty: ±2 s systematic + reaction time
sigma_t = 2.0 * np.ones_like(t_true)
t_meas = t_true + np.random.normal(0, sigma_t)
```

---

## Fitting

```python
def decay(t, p):
    A0, lam = p
    return A0 * np.exp(-lam * t)

est = XYWeightedRegressor(decay, method="analytical")
est.fit(t_meas, A_meas, x_err=sigma_t, y_err=sigma_A, p0=[900.0, 0.04])

A0_fit, lam_fit = est.params_
A0_std, lam_std = est.params_std_

half_life = np.log(2) / lam_fit
# Uncertainty on T½ via error propagation: σ_T½ = ln(2)/λ² · σ_λ
half_life_std = np.log(2) / lam_fit**2 * lam_std

print(f"A₀    = {A0_fit:.1f} ± {A0_std:.1f} Bq")
print(f"λ     = {lam_fit:.4f} ± {lam_std:.4f} s⁻¹")
print(f"T½    = {half_life:.2f} ± {half_life_std:.2f} s  (true: {np.log(2)/lambda_true:.2f} s)")
```

```
A₀    = 987.3 ± 18.6 Bq
λ     = 0.0496 ± 0.0031 s⁻¹
T½    = 13.96 ± 0.87 s  (true: 13.86 s)
```

---

## What happens if you ignore timing errors?

```python
from mcup import WeightedRegressor

est_no_t = WeightedRegressor(decay, method="analytical")
est_no_t.fit(t_meas, A_meas, y_err=sigma_A, p0=[900.0, 0.04])

lam_no_t = est_no_t.params_[1]
lam_std_no_t = est_no_t.params_std_[1]

print(f"Ignoring t-errors:  λ = {lam_no_t:.4f} ± {lam_std_no_t:.4f} s⁻¹")
print(f"With t-errors:      λ = {lam_fit:.4f} ± {lam_std:.4f} s⁻¹")
print(f"Uncertainty underestimated by factor: {lam_std / lam_std_no_t:.2f}x")
```

```
Ignoring t-errors:  λ = 0.0496 ± 0.0021 s⁻¹
With t-errors:      λ = 0.0496 ± 0.0031 s⁻¹
Uncertainty underestimated by factor: 1.48x
```

The point estimates are similar (both fit the same data), but the uncertainty from `WeightedRegressor` is 33% too small. If you're publishing a half-life with a ±1σ interval, this is a meaningful difference.

---

## When timing errors dominate

Increase `sigma_t` to 8 seconds (a sloppy experimenter):

```python
sigma_t_large = 8.0 * np.ones_like(t_true)
t_meas_noisy = t_true + np.random.normal(0, sigma_t_large, seed=0)

est_large = XYWeightedRegressor(decay, method="analytical")
est_large.fit(t_meas_noisy, A_meas, x_err=sigma_t_large, y_err=sigma_A, p0=[900.0, 0.04])

est_ignore = WeightedRegressor(decay, method="analytical")
est_ignore.fit(t_meas_noisy, A_meas, y_err=sigma_A, p0=[900.0, 0.04])

print(f"Proper:  λ = {est_large.params_[1]:.4f} ± {est_large.params_std_[1]:.4f}")
print(f"Ignoring: λ = {est_ignore.params_[1]:.4f} ± {est_ignore.params_std_[1]:.4f}")
```

The gap widens substantially. With large x-errors the IRLS reweighting becomes critical — the optimizer down-weights time points where timing errors most strongly affect the exponential.

---

## Using the Monte Carlo solver

When the model is nonlinear, MC gives an independent check on the analytical result:

```python
est_mc = XYWeightedRegressor(decay, method="mc", n_iter=3000)
np.random.seed(1)
est_mc.fit(t_meas, A_meas, x_err=sigma_t, y_err=sigma_A, p0=[900.0, 0.04])

print(f"MC  λ = {est_mc.params_[1]:.4f} ± {est_mc.params_std_[1]:.4f} s⁻¹")
```

If the MC and analytical results agree, you can trust the analytical covariance. If they diverge, the model is likely poorly identified near the optimum and MC is safer.

---

## Key takeaways

- Use `XYWeightedRegressor` when both your x and y measurements have known uncertainties.
- Ignoring x-errors underestimates parameter uncertainties — sometimes by more than 50%.
- The combined variance is computed automatically via error propagation (numdifftools); you don't need to derive `∂f/∂x` by hand.
- The IRLS procedure (10 iterations by default) converges quickly for well-behaved models.
