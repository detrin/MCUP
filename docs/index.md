# MCUP

**Monte Carlo Uncertainty Propagation** — a Python library for regression with measurement errors.

MCUP provides three sklearn-like estimators that correctly propagate x and y measurement uncertainties into parameter confidence intervals.

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/mcup.svg)](https://pypi.org/project/mcup/)
[![PyPI version](https://img.shields.io/pypi/v/mcup.svg)](https://pypi.org/project/mcup/)
[![CI](https://github.com/detrin/MCUP/actions/workflows/package-main.yml/badge.svg)](https://github.com/detrin/MCUP/actions/workflows/package-main.yml)
[![Documentation Status](https://readthedocs.org/projects/mcup/badge/?version=latest)](https://readthedocs.org/projects/mcup/?badge=latest)
[![codecov](https://codecov.io/gh/detrin/MCUP/branch/master/graph/badge.svg?token=Dx6elQkztR)](https://codecov.io/gh/detrin/MCUP)

---

## Estimators

| Estimator | Use when | Error model |
|-----------|----------|-------------|
| `WeightedRegressor` | Only y has measurement errors | `Σ (y - f(x))² / σ_y²` |
| `XYWeightedRegressor` | Both x and y have errors | Combined variance via error propagation |
| `DemingRegressor` | Both x and y have errors (exact) | Joint optimisation over parameters + latent true x |

Each estimator supports two solvers via `method`:

- `"analytical"` — weighted least squares + `(J^T W J)^{-1}` covariance (fast)
- `"mc"` — Monte Carlo sampling with weighted objective + Welford covariance (robust for nonlinear models)

---

## Installing

```bash
python3 -m pip install mcup
```

**Dev version:**

```bash
python3 -m pip install git+https://github.com/detrin/MCUP.git#egg=mcup
```

---

## Quick Example

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

See the [Getting Started guide](guide/getting-started.md) for more examples, or the [API Reference](api/weighted.md) for full documentation.
