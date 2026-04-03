# MCUP

MCUP (Monte Carlo Uncertainty Propagation) is a Python library for regression with measurement errors. It provides three sklearn-like estimators that correctly propagate x and y measurement uncertainties into parameter confidence intervals.

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/mcup.svg)](https://pypi.org/project/mcup/) [![PyPI version](https://img.shields.io/pypi/v/mcup.svg)](https://pypi.org/project/mcup/) [![CI](https://github.com/detrin/MCUP/actions/workflows/package-main.yml/badge.svg)](https://github.com/detrin/MCUP/actions/workflows/package-main.yml) [![codecov](https://codecov.io/gh/detrin/MCUP/branch/master/graph/badge.svg?token=Dx6elQkztR)](https://codecov.io/gh/detrin/MCUP)

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

Using the wrong estimator (OLS when x has errors) breaks coverage:

| Scenario | Wrong estimator | Coverage | Correct estimator | Coverage |
|----------|-----------------|----------|-------------------|----------|
| Exp decay + timing errors | WeightedRegressor | ✗ 30% | XYWeightedRegressor | ✓ 64% |
| Beer-Lambert | WeightedRegressor | ✗ 7% | XYWeightedRegressor | ✓ 68% |
| Method comparison | WeightedRegressor (OLS) | ✗ 32% | DemingRegressor | ✓ 66% |

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

A photodetector where noise grows with signal: points at high intensity are less reliable. OLS doesn't know that and gives overconfident slope uncertainty. `WeightedRegressor` down-weights noisy points and produces calibrated intervals.

```python
import numpy as np
from mcup import WeightedRegressor

rng = np.random.default_rng(42)
x = np.linspace(1, 10, 30)
y_err = 0.1 * x                  # noise grows with x
y = 2.0 * x + 1.0 + rng.normal(0, y_err)

def line(x, p):
    return p[0] + p[1] * x

# Uniform weights (wrong — ignores that high-x points are noisier)
ols = WeightedRegressor(line, method="analytical")
ols.fit(x, y, y_err=np.ones_like(x), p0=[0.0, 1.0])

# Correct weights from measurement errors
wls = WeightedRegressor(line, method="analytical")
wls.fit(x, y, y_err=y_err, p0=[0.0, 1.0])

print(f"OLS:      slope = {ols.params_[1]:.3f} ± {ols.params_std_[1]:.4f}  ← overconfident")
print(f"Weighted: slope = {wls.params_[1]:.3f} ± {wls.params_std_[1]:.4f}  ← calibrated")
# true slope = 2.0
```

**Case 2 — both x and y have errors**

A spring balance where both extension (x) and force (y) are measured with error. Ignoring x-errors causes attenuation bias (slope pulled toward zero) and intervals that are far too narrow. `XYWeightedRegressor` propagates both error sources.

```python
from mcup import XYWeightedRegressor

rng = np.random.default_rng(0)
x_true = np.linspace(0.1, 2.0, 25)
x_err, y_err = 0.05 * np.ones(25), 0.15 * np.ones(25)
x_obs = x_true + rng.normal(0, x_err)
y = 8.0 * x_true + rng.normal(0, y_err)   # true spring constant k=8

# Ignoring x-errors (wrong)
bad = WeightedRegressor(line, method="analytical")
bad.fit(x_obs, y, y_err=y_err, p0=[0.0, 1.0])

# Propagating both errors (correct)
est = XYWeightedRegressor(line, method="analytical")
est.fit(x_obs, y, x_err=x_err, y_err=y_err, p0=[0.0, 1.0])

print(f"Ignoring x-err: k = {bad.params_[1]:.3f} ± {bad.params_std_[1]:.3f}  ← biased low, too narrow")
print(f"XYWeighted:     k = {est.params_[1]:.3f} ± {est.params_std_[1]:.3f}  ← unbiased, calibrated")
# true k = 8.0
```

See [DEVELOPING.md](DEVELOPING.md) for contributing, running tests, and building docs.
