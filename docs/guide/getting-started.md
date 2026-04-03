# Getting Started

## Installation

=== "uv"
    ```bash
    uv add mcup
    ```

=== "pip"
    ```bash
    pip install mcup
    ```

**Latest from master:**

=== "uv"
    ```bash
    uv add git+https://github.com/detrin/MCUP.git
    ```

=== "pip"
    ```bash
    pip install git+https://github.com/detrin/MCUP.git
    ```

---

## Examples

### y-errors only — WeightedRegressor (analytical)

Use this when only your y measurements carry uncertainty.

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

---

### x and y errors — XYWeightedRegressor (combined variance, IRLS)

Use this when both x and y carry measurement errors and your model is mildly nonlinear.
The combined variance is computed via error propagation and the fit is iterated (IRLS).

```python
import numpy as np
from mcup import XYWeightedRegressor

def line(x, p):
    return p[0] + p[1] * x

x = np.linspace(0, 10, 30)
y = line(x, [1.0, 2.0]) + np.random.normal(0, 0.5, 30)
x_err = 0.3 * np.ones(30)
y_err = 0.5 * np.ones(30)

est = XYWeightedRegressor(line, method="analytical")
est.fit(x, y, x_err=x_err, y_err=y_err, p0=[0.0, 0.0])

print(est.params_)
print(est.params_std_)
```

---

### x and y errors — DemingRegressor (exact joint optimisation)

Use this when you need the most rigorous treatment of x and y errors.
The optimisation runs jointly over model parameters and latent true x values.

```python
import numpy as np
from mcup import DemingRegressor

def line(x, p):
    return p[0] + p[1] * x

x = np.linspace(0, 10, 30)
y = line(x, [1.0, 2.0]) + np.random.normal(0, 0.5, 30)
x_err = 0.3 * np.ones(30)
y_err = 0.5 * np.ones(30)

est = DemingRegressor(line, method="analytical")
est.fit(x, y, x_err=x_err, y_err=y_err, p0=[0.0, 0.0])

print(est.params_)
print(est.params_std_)
```

---

### Monte Carlo solver — works for any nonlinear model

Switch any estimator to `method="mc"` for Monte Carlo uncertainty propagation.
This is more robust for strongly nonlinear models where the analytical Jacobian
approximation breaks down.

```python
import numpy as np
from mcup import WeightedRegressor

def exponential(x, p):
    return p[0] * np.exp(p[1] * x)

x = np.linspace(0, 10, 30)
y = exponential(x, [1.0, 0.3]) + np.random.normal(0, 0.5, 30)
y_err = 0.5 * np.ones(30)

est = WeightedRegressor(exponential, method="mc", n_iter=5000)
est.fit(x, y, y_err=y_err, p0=[1.0, 0.1])

print(est.params_)
print(est.params_std_)
print(est.n_iter_)  # actual iterations run
```

---

### Convergence-based stopping (MC only)

Instead of a fixed iteration count, you can let the MC solver stop automatically
once parameter estimates have converged to within `rtol` (relative) or `atol`
(absolute) tolerance.

```python
import numpy as np
from mcup import WeightedRegressor

def line(x, p):
    return p[0] + p[1] * x

x = np.linspace(0, 10, 30)
y = line(x, [1.0, 2.0]) + np.random.normal(0, 0.5, 30)
y_err = 0.5 * np.ones(30)

est = WeightedRegressor(line, method="mc", rtol=1e-4, atol=1e-4)
est.fit(x, y, y_err=y_err, p0=[0.0, 0.0])
print(f"Converged after {est.n_iter_} iterations")
```

---

## sklearn compatibility

All estimators implement `get_params()` and `set_params()` for use with sklearn pipelines and grid search.
