# Changelog

## 1.1.0

- Multivariable X support: all three estimators now accept `X` of shape `(n, k)`; `func(x, params)` receives `X[i]` directly
- Zero-error feature handling: set `x_err[:, j] = 0` for controlled/exact variables — MC won't perturb that column, Deming pins latent values to observed
- Tutorial 4: multivariable regression (heat output model `P = a·I² + b·T + c`) with mixed-error inputs
- Performance guide: analytical vs Monte Carlo runtime comparison with figures
- More dramatic tutorial examples: sensor calibration (9:1 noise ratio, 5.4× confidence band difference), radioactive decay (σ_t=15 s, +32% bias + 3× underestimated uncertainty), method comparison (σ_x=25 μg/mL, −19% attenuation bias)

## 1.0.0

- Type hints throughout the codebase (`from __future__ import annotations`, full mypy coverage)
- Ruff linting (E, W, F, I, B rules) added to pre-commit hook alongside mypy
- Replaced black/flake8 with ruff in dev dependencies
- Benchmark suite: 13 physical scenarios, 200 parameter samples each, bias/RMSE/coverage statistics
- Method comparison page in docs with figures and benchmark tables
- Tutorials: sensor calibration, radioactive decay, Deming method comparison
- MkDocs Material documentation with API reference via mkdocstrings

## 0.2.0

- Redesigned API: three sklearn-like estimators (WeightedRegressor, XYWeightedRegressor, DemingRegressor)
- Both analytical and Monte Carlo solvers for all estimators
- Welford online covariance tracking for MC solver
- Full covariance matrix output

## 0.1.1

- Initial release: Monte Carlo parameter error estimator (PEE)
