# MCUP Redesign — Design Spec
**Date:** 2026-04-03  
**Version target:** 0.2.0

---

## Goal

Extend MCUP from a Monte Carlo-only utility into a full measurement-error regression library that implements all statistically rigorous approaches from the companion notebook ("Measurement error in regression"). Redesign the public API to follow sklearn conventions.

---

## Regression Cases

Three estimators cover the cases from the notebook (case 1 — no errors — is excluded as it is plain scipy):

| Case | Estimator | Cost function |
|------|-----------|---------------|
| y-errors only | `WeightedRegressor` | `Σ (y - f(x))² / σ_y²` |
| y + x errors | `XYWeightedRegressor` | `Σ (y - f(x))² / (σ_y² + Σ βₖ²σ_xₖ²)` — βₖ evaluated at current parameter estimate (iteratively reweighted) |
| Deming regression | `DemingRegressor` | `Σ [(xᵢ - ηᵢ)²/σ_xᵢ² + (yᵢ - f(ηᵢ))²/σ_yᵢ²]` over β + η |

---

## Architecture

```
mcup/
├── __init__.py          # exports WeightedRegressor, XYWeightedRegressor, DemingRegressor
├── base.py              # BaseRegressor: sklearn contract, shared fit logic
├── weighted.py          # WeightedRegressor
├── xy_weighted.py       # XYWeightedRegressor
├── deming.py            # DemingRegressor
├── _analytical.py       # private: weighted LS + (J^T W J)^{-1} covariance
├── _mc.py               # private: MC sampler + weighted objective + Welford
├── _utils.py            # Welford, local_numpy_seed (from current utils.py)
└── data_generator.py    # unchanged
```

`BaseRegressor` holds the sklearn contract. Each estimator inherits it and overrides only cost function construction and weight matrix logic. `_analytical.py` and `_mc.py` are private solvers — not part of the public API.

---

## Public API

### Construction

```python
est = WeightedRegressor(
    func,                     # callable: f(x, *params) -> y
    method="mc",              # "mc" | "analytical"
    n_iter=10_000,            # MC only: fixed iteration count
    rtol=None,                # MC only: relative tolerance (replaces n_iter if set)
    atol=None,                # MC only: absolute tolerance (replaces n_iter if set)
    optimizer="Nelder-Mead",  # scipy optimizer
)
# XYWeightedRegressor and DemingRegressor: same signature
```

### Fit

```python
est.fit(X, y, y_err=ye)                    # WeightedRegressor
est.fit(X, y, x_err=xe, y_err=ye)          # XYWeightedRegressor
est.fit(X, y, x_err=xe, y_err=ye)          # DemingRegressor
```

### Fitted attributes

```python
est.params_       # ndarray — parameter means
est.params_std_   # ndarray — std (MC: empirical; analytical: sqrt(diag(cov)))
est.covariance_   # ndarray — full covariance matrix
est.n_iter_       # int — MC only: actual iterations run
```

### sklearn compatibility

`BaseRegressor` implements `get_params()` and `set_params()` per sklearn convention, enabling use in `Pipeline` and `GridSearchCV`.

---

## Solvers

### Analytical (`_analytical.py`)

1. Construct weight matrix W from error arrays
2. Minimize weighted cost via `scipy.optimize.minimize`
3. Compute Jacobian at solution via `numdifftools.Jacobian`
4. Return covariance = `(J^T W J)^{-1}`

### Monte Carlo (`_mc.py`)

1. For each iteration:
   - Sample `x' ~ N(x, σ_x)`, `y' ~ N(y, σ_y)`
   - Minimize **weighted** cost on perturbed data
   - Update parameter mean and covariance via Welford's online algorithm
2. Terminate at `n_iter` or when `rtol`/`atol` convergence criteria met on both mean and std
3. Return empirical mean and covariance

**Key fix vs 0.1.1:** MC now uses the weighted objective (same weights as analytical) rather than unweighted L2 norm.

### DemingRegressor specifics

Optimization variable includes both β (model parameters) and η (latent true x values). Initial η = observed x. Solver is always `scipy.optimize.minimize`; analytical method computes `(J^T W J)^{-1}` over the full β+η space then returns the top-left `len(β) × len(β)` block as the β covariance.

---

## Testing

One test file per estimator plus solver unit tests:

```
tests/
├── test_weighted.py
├── test_xy_weighted.py
├── test_deming.py
├── test_analytical.py
├── test_mc.py
└── test_data_generator.py
```

Each test:
- Uses `DataGenerator` with fixed seed (42) for reproducible synthetic data
- Fits both `method="mc"` and `method="analytical"`
- Asserts `params_` within tolerance of ground truth parameters
- Asserts `params_std_` is positive and finite

---

## Backwards Compatibility

`Measurement` class and `parameter_error_estimator()` are removed entirely. No deprecation shims. This is a breaking change: **0.1.1 → 0.2.0**.

---

## Documentation

One example notebook per estimator under `docs/examples/`:
- `weighted_regressor.ipynb`
- `xy_weighted_regressor.ipynb`
- `deming_regressor.ipynb`

Each example shows: sklearn-like workflow, analytical vs MC comparison, parameter confidence intervals.
