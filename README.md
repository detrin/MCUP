# MCUP
MCUP (Monte Carlo Uncertainty Propagation) is a Python library for regression with measurement errors. It provides three sklearn-like estimators that correctly propagate x and y measurement uncertainties into parameter confidence intervals.

## Status
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/mcup.svg)](https://pypi.org/project/mcup/) [![PyPI version shields.io](https://img.shields.io/pypi/v/mcup.svg)](https://pypi.org/project/mcup/) [![master](https://github.com/detrin/MCUP/actions/workflows/package-main.yml/badge.svg)](https://github.com/detrin/MCUP/actions/workflows/package-main.yml) [![Documentation Status](https://readthedocs.org/projects/mcup/badge/?version=latest)](https://readthedocs.org/projects/mcup/?badge=latest) [![codecov](https://codecov.io/gh/detrin/MCUP/branch/master/graph/badge.svg?token=Dx6elQkztR)](https://codecov.io/gh/detrin/MCUP)

## Estimators

| Estimator | Use when | Error model |
|-----------|----------|-------------|
| `WeightedRegressor` | Only y has measurement errors | `Σ (y - f(x))² / σ_y²` |
| `XYWeightedRegressor` | Both x and y have errors | Combined variance via error propagation |
| `DemingRegressor` | Both x and y have errors (exact) | Joint optimization over parameters + latent true x |

Each estimator supports two solvers via `method`:
- `"analytical"` — weighted least squares + `(J^T W J)^{-1}` covariance (fast)
- `"mc"` — Monte Carlo sampling with weighted objective + Welford covariance (robust for nonlinear models)

## Installing MCUP

```bash
python3 -m pip install mcup
```

#### Dev Version

```bash
python3 -m pip install git+https://github.com/detrin/MCUP.git#egg=mcup
```

## Examples

### y-errors only

```python
import numpy as np
from mcup import WeightedRegressor

def line(x, p):
    return p[0] + p[1] * x

x = np.linspace(0, 10, 30)
y = line(x, [1.0, 2.0]) + np.random.normal(0, 0.5, 30)
y_err = 0.5 * np.ones(30)

est = WeightedRegressor(line, method="analytical")
est.fit(x, y, y_err=y_err, p0=[0.0, 0.0])

print(est.params_)      # [~1.0, ~2.0]
print(est.params_std_)  # parameter uncertainties
print(est.covariance_)  # full covariance matrix
```

### x and y errors — combined variance (IRLS)

```python
from mcup import XYWeightedRegressor

x_err = 0.3 * np.ones(30)

est = XYWeightedRegressor(line, method="analytical")
est.fit(x, y, x_err=x_err, y_err=y_err, p0=[0.0, 0.0])

print(est.params_)
print(est.params_std_)
```

### x and y errors — Deming regression (exact joint optimization)

```python
from mcup import DemingRegressor

est = DemingRegressor(line, method="analytical")
est.fit(x, y, x_err=x_err, y_err=y_err, p0=[0.0, 0.0])

print(est.params_)
print(est.params_std_)
```

### Monte Carlo solver (works for any nonlinear model)

```python
from mcup import WeightedRegressor

def exponential(x, p):
    return p[0] * np.exp(p[1] * x)

est = WeightedRegressor(exponential, method="mc", n_iter=5000)
est.fit(x, y, y_err=y_err, p0=[1.0, 0.1])

print(est.params_)
print(est.params_std_)
print(est.n_iter_)  # actual iterations run
```

### Convergence-based stopping (MC only)

```python
est = WeightedRegressor(line, method="mc", rtol=1e-4, atol=1e-4)
est.fit(x, y, y_err=y_err, p0=[0.0, 0.0])
print(f"Converged after {est.n_iter_} iterations")
```

## sklearn compatibility

All estimators implement `get_params()` and `set_params()` for use with sklearn pipelines.

## Contributing
When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change.
