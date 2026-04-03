# Developing MCUP

## Setup

```bash
git clone https://github.com/detrin/MCUP.git
cd MCUP
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
```

## Project structure

```
mcup/
├── weighted.py       # WeightedRegressor  (y-errors only)
├── xy_weighted.py    # XYWeightedRegressor (x+y errors, IRLS)
├── deming.py         # DemingRegressor    (x+y errors, joint optimization)
├── base.py           # BaseRegressor      (sklearn interface)
├── _analytical.py    # analytical solver  (private)
├── _mc.py            # Monte Carlo solver (private)
├── _utils.py         # Welford + seed     (private)
└── data_generator.py # synthetic data for tests
tests/
docs/
```

## Common tasks

```bash
make test        # run pytest with coverage
make lint        # flake8 + black check
make format      # black autoformat
make docs        # build MkDocs site to site/
make docs_serve  # live-reload docs at localhost:8000
make build       # build sdist + wheel to dist/
make clean       # remove build artefacts
```

## Running tests

```bash
pytest                          # all tests
pytest tests/test_weighted.py   # single file
pytest -k "analytical"          # filter by name
pytest --no-cov                 # skip coverage
```

## Building and releasing

```bash
make build                  # produces dist/mcup-X.Y.Z.*
make upload_test            # push to test.pypi.org
make upload                 # push to pypi.org
```

Requires a `~/.pypirc` with credentials or `TWINE_USERNAME` / `TWINE_PASSWORD` env vars.

## Adding a new estimator

1. Create `mcup/your_estimator.py` — subclass `BaseRegressor`, override `fit()`
2. Add tests in `tests/test_your_estimator.py`
3. Export from `mcup/__init__.py`
4. Add an API page under `docs/api/`
5. Update `docs/guide/choosing-estimator.md`

## Before opening a PR

- `make lint` passes
- `make test` passes with no regressions
- New code has test coverage
- Discuss the change in an issue first
