MODULE_NAME = mcup
MODULE_VERSION = 0.2.0

.PHONY: install
install:
	pip install -e ".[dev,docs]"

.PHONY: test
test:
	pytest

.PHONY: lint
lint:
	flake8 mcup tests --count --exit-zero --max-line-length=100 --statistics
	black --check mcup tests

.PHONY: format
format:
	black mcup tests

.PHONY: docs
docs:
	mkdocs build

.PHONY: docs_serve
docs_serve:
	mkdocs serve

.PHONY: build
build:
	python -m build

.PHONY: upload
upload:
	twine upload dist/$(MODULE_NAME)-$(MODULE_VERSION)*

.PHONY: upload_test
upload_test:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/$(MODULE_NAME)-$(MODULE_VERSION)*

.PHONY: clean
clean:
	rm -rf dist *.egg-info htmlcov site .coverage
	find . -type d -name '__pycache__' | xargs rm -rf
	find . -name '*.pyc' | xargs rm -rf
	find . -name '.pytest_cache' | xargs rm -rf
