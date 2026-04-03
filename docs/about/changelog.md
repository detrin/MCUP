# Changelog

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
