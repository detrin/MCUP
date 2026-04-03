# MCUP

MCUP (Monte Carlo Uncertainty Propagation) is a Python library for regression with measurement errors. It provides three sklearn-like estimators that correctly propagate x and y measurement uncertainties into parameter confidence intervals.

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/mcup.svg)](https://pypi.org/project/mcup/) [![PyPI version](https://img.shields.io/pypi/v/mcup.svg)](https://pypi.org/project/mcup/) [![CI](https://github.com/detrin/MCUP/actions/workflows/package-main.yml/badge.svg)](https://github.com/detrin/MCUP/actions/workflows/package-main.yml) [![codecov](https://codecov.io/gh/detrin/MCUP/branch/master/graph/badge.svg?token=Dx6elQkztR)](https://codecov.io/gh/detrin/MCUP)

## Install

```bash
uv add mcup
```

Or with pip:

```bash
pip install mcup
```

## Quick start

The core idea: you have data where measurement noise is not uniform, or x itself is measured. MCUP gives you honest parameter uncertainties in both cases.

**Case 1 — only y has errors (heteroscedastic noise)**

A voltage sensor where ADC noise grows with signal: σ = 0.01 + 0.08·V — low-voltage points (σ ≈ 0.05 V) are 9× more reliable than high-voltage points (σ ≈ 0.41 V). OLS treats them all equally and reports an uncertainty **5× too wide**.

```python
import numpy as np
from mcup import WeightedRegressor

np.random.seed(42)
V_ref = np.linspace(0.5, 5.0, 30)
sigma = 0.01 + 0.08 * V_ref          # 9:1 noise ratio (0.05 V → 0.41 V)
V_sensor = 0.05 + 1.02 * V_ref + np.random.normal(0, sigma)

def line(x, p):
    return p[0] + p[1] * x

# OLS: treats all 30 points as equally noisy (wrong)
ols = WeightedRegressor(line, method="analytical")
ols.fit(V_ref, V_sensor, y_err=np.ones_like(V_ref), p0=[0.0, 1.0])

# Weighted: down-weights the noisy high-voltage points (correct)
wls = WeightedRegressor(line, method="analytical")
wls.fit(V_ref, V_sensor, y_err=sigma, p0=[0.0, 1.0])

print(f"OLS:      gain b = {ols.params_[1]:.3f} ± {ols.params_std_[1]:.3f}  ← 5× too wide")
print(f"Weighted: gain b = {wls.params_[1]:.3f} ± {wls.params_std_[1]:.3f}  ← correct")
# OLS:      gain b = 0.970 ± 0.136  ← 5× too wide
# Weighted: gain b = 0.967 ± 0.025  ← correct
# true b = 1.02
```

**Case 2 — both x and y have errors (bias + underestimated uncertainty)**

Radioactive decay where timing jitter is ±15 s. At early time points (t = 5 s), ±15 s represents 300% of the elapsed time. Ignoring x-errors biases λ by +32% **and** shrinks σ_λ by 3×:

```python
from mcup import XYWeightedRegressor

np.random.seed(7)
t_true = np.array([0, 5, 10, 20, 35, 50, 70, 90, 120, 150], dtype=float)
A_true = 1000.0 * np.exp(-0.05 * t_true)
sigma_A = np.sqrt(A_true)
sigma_t = 15.0 * np.ones_like(t_true)     # ±15 s timing jitter
t_meas = np.maximum(t_true + np.random.normal(0, sigma_t), 0.0)
A_meas = np.clip(A_true + np.random.normal(0, sigma_A), 1.0, None)

def decay(t, p):
    return p[0] * np.exp(-p[1] * t)

# Ignoring x-errors (wrong): biased estimate, overconfident interval
bad = WeightedRegressor(decay, method="analytical")
bad.fit(t_meas, A_meas, y_err=sigma_A, p0=[900.0, 0.04])

# Propagating both errors (correct)
est = XYWeightedRegressor(decay, method="analytical")
est.fit(t_meas, A_meas, x_err=sigma_t, y_err=sigma_A, p0=[900.0, 0.04])

print(f"Ignoring t-errors: λ = {bad.params_[1]:.4f} ± {bad.params_std_[1]:.4f}  ← +32% bias, 3× too narrow")
print(f"XYWeighted:        λ = {est.params_[1]:.4f} ± {est.params_std_[1]:.4f}  ← unbiased, honest")
# Ignoring t-errors: λ = 0.0658 ± 0.0018  ← +32% bias, 3× too narrow
# XYWeighted:        λ = 0.0489 ± 0.0057  ← unbiased, honest
# true λ = 0.05
```

## Why MCUP

Standard least squares (OLS) assumes all observations are equally reliable. Real experiments break this in two common ways:

- **Heteroscedastic y-errors** — measurement noise varies across the range. OLS overweights noisy points, biasing the fit and producing overconfident uncertainties.
- **Errors in x** — when the independent variable is itself measured (time, concentration, displacement), ignoring those errors causes attenuation bias: slopes are pulled toward zero, and uncertainty intervals shrink below their true size.

**Why not just use the covariance matrix from the optimizer?**

When measurement errors are large, the standard approach of reading off `sqrt(diag(cov))` from the fit residuals underestimates the true parameter uncertainty. The covariance matrix tells you how well the optimizer converged — it does not propagate the uncertainty that came *in* with your data. MCUP propagates measurement noise directly through the model so that `params_std_` reflects both fit quality and input uncertainty. For a worked example comparing both approaches, see this [Kaggle notebook on measurement error in regression](https://www.kaggle.com/code/jetakow/measurement-error-in-regression).

MCUP fixes both problems. The figure below illustrates the effect for a linear calibration with heteroscedastic y-errors:

![Comparison of OLS vs WeightedRegressor](docs/assets/comparison_linear.png)

*Left: OLS (red) fits the same data differently from weighted regression (blue) because it treats all points equally regardless of σ_y. Right: over 500 simulated experiments, OLS coverage deviates from the nominal 68.3% — WeightedRegressor stays calibrated.*

## Estimators

| Estimator | Use when | Error model |
|-----------|----------|-------------|
| `WeightedRegressor` | Only y has measurement errors | `Σ (y − f(x))² / σ_y²` |
| `XYWeightedRegressor` | Both x and y have errors (nonlinear) | Combined variance via error propagation (IRLS) |
| `DemingRegressor` | Both x and y have errors (linear only) | Joint optimisation over parameters + latent true x |

Each estimator supports:
- `method="analytical"` — weighted LS + `(J^T W J)^{-1}` covariance (fast, exact for well-posed problems)
- `method="mc"` — Monte Carlo with Welford covariance (robust cross-check for nonlinear models)

## Benchmark summary

Validated across 13 physical scenarios (200 independent parameter configurations each). The analytical solver achieves well-calibrated 1σ uncertainty intervals on all scenarios:

| Scenario | Estimator | Bias | RMSE | Coverage |
|----------|-----------|------|------|----------|
| Linear calibration (homo) | WeightedRegressor | +0.3% | 12.8% | ✓ 68% |
| Linear calibration (hetero) | WeightedRegressor | +0.5% | 7.2% | ✓ 71% |
| Radioactive decay | WeightedRegressor | −0.0% | 2.6% | ✓ 64% |
| Power law (diffusion) | WeightedRegressor | +0.0% | 4.6% | ✓ 68% |
| Gaussian spectral peak | WeightedRegressor | −0.1% | 1.7% | ✓ 66% |
| Damped oscillator | WeightedRegressor | −0.4% | 7.2% | ✓ 67% |
| Exp decay + timing errors | **XYWeightedRegressor** | −1.2% | 5.0% | ✓ 64% |
| Hooke's law (x+y errors) | **XYWeightedRegressor** | −1.0% | 54% | ✓ 75% |
| Beer-Lambert (x+y errors) | **XYWeightedRegressor** | +46% | 220% | ✓ 68% |
| Method comparison | **DemingRegressor** | +14% | 111% | ✓ 64% |
| Isotope ratio MS | **DemingRegressor** | +3.2% | 420% | ✓ 72% |
| Small sample (n=8) | WeightedRegressor | −2.7% | 29% | ✓ 69% |
| Low SNR | WeightedRegressor | −1.9% | 136% | ✓ 67% |

Bias and RMSE are relative to the true parameter values. Large RMSE on near-zero intercepts (Beer-Lambert baseline, isotope intercept) reflects small absolute values — the coverage column is the reliable calibration metric.

### OLS baseline: when ignoring measurement errors breaks uncertainty estimation

Plain OLS (no error weighting) estimates parameter uncertainty from fit residuals alone — `σ² = SSR/(n−p)`. This works when noise is truly uniform. When noise varies across the range, OLS produces miscalibrated intervals even though the parameter estimates themselves may look reasonable.

| Scenario | OLS coverage | WeightedRegressor coverage | What goes wrong |
|----------|:------------:|:--------------------------:|-----------------|
| S1 Linear (homo σ_y=0.5) | ✓ 68%/70% | ✓ 68%/70% | — OLS works; noise is uniform |
| S2 Linear (hetero σ_y=0.1+0.1·x) | ~ 86%/72% | ✓ 71%/72% | Intervals too wide; pooled σ² inflated by noisy high-x points |
| S3 Radioactive decay (Poisson √A) | ✗ 32%/42% | ✓ 64%/68% | Badly overconfident; large early-time counts dominate residuals |
| S4 Power law (8% relative noise) | ✓ 66%/66% | ✓ 68%/69% | — OLS approximately ok here |
| S5 Gaussian peak (Poisson counts) | ✗ 39%/54% | ✓ 66%/70% | Overconfident; amplitude and center poorly constrained |
| S6 Damped oscillator (uniform σ_y) | ✓ 64%/71% | ✓ 67%/72% | — OLS works; noise is uniform |

**The pattern:** OLS coverage is correct only when σ_y is constant across the range (S1, S6). As soon as noise scales with signal — Poisson counting (S3, S5) or percentage-of-reading errors (S2, S4) — the pooled residual variance is a poor proxy for per-point noise, and uncertainty intervals become unreliable. The parameter estimates themselves are often similar; it is the *uncertainty* that OLS gets wrong.

### Using the wrong estimator when x has errors

| Scenario | Wrong estimator | Coverage | Correct estimator | Coverage |
|----------|-----------------|:--------:|-------------------|:--------:|
| Exp decay + timing errors | WeightedRegressor | ✗ 30% | XYWeightedRegressor | ✓ 64% |
| Beer-Lambert | WeightedRegressor | ✗ 7% | XYWeightedRegressor | ✓ 68% |
| Method comparison | WeightedRegressor (OLS) | ✗ 32% | DemingRegressor | ✓ 66% |

See [DEVELOPING.md](DEVELOPING.md) for contributing, running tests, and building docs.
