# MCUP

MCUP (Monte Carlo Uncertainty Propagation) is a Python library for regression with measurement errors. It provides three sklearn-like estimators that correctly propagate x and y measurement uncertainties into parameter confidence intervals.

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/mcup.svg)](https://pypi.org/project/mcup/) [![PyPI version](https://img.shields.io/pypi/v/mcup.svg)](https://pypi.org/project/mcup/) [![CI](https://github.com/detrin/MCUP/actions/workflows/package-main.yml/badge.svg)](https://github.com/detrin/MCUP/actions/workflows/package-main.yml) [![codecov](https://codecov.io/gh/detrin/MCUP/branch/master/graph/badge.svg?token=Dx6elQkztR)](https://codecov.io/gh/detrin/MCUP)

## Estimators

| Estimator | Use when | Method |
|-----------|----------|--------|
| `WeightedRegressor` | Only y has measurement errors | Weighted LS, `Σ (y−f(x))²/σ_y²` |
| `XYWeightedRegressor` | Both x and y have errors | Combined variance via error propagation (IRLS) |
| `DemingRegressor` | Both x and y have errors (exact) | Joint optimization over parameters + latent true x |

Each estimator supports `method="analytical"` (fast, `(J^T W J)^{-1}` covariance) and `method="mc"` (Monte Carlo, robust for nonlinear models).

## Install

```bash
pip install mcup
```

## Quick start

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

See [DEVELOPING.md](DEVELOPING.md) for how to contribute, run tests, and build docs.
