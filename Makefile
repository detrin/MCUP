##############################################################
#
# MCUP Makefile
#
# This file defines several tasks useful to MCUP devs
#
##############################################################

# There should be no need to change this,
# if you do you'll also need to update docker-compose.yml
SERVICE_TARGET := mcup

PWD ?= pwd_unknown

# retrieve NAME from /variables file
MODULE_NAME = mcup
MODULE_VERSION = 0.1.1
MODULE_TEST_VERSION = 0.1.10

# cli prefix for commands to run in container
RUN_DOCK = \
	docker-compose -p mcup run --rm mcup /bin/sh -l -c


.PHONY: docker_build 
docker_build:
	docker-compose build

.PHONY: docker_rm 
docker_rm:
	yes | docker-compose down

.PHONY: shell 
shell:
	$(RUN_DOCK) "pip install -r requirements.txt \
		&& bash"

.PHONY: module
module: 
	@# ensure there is a symlink from MODULE_NAME to module directory
	@# then run regular setup.py to build the module
	$(RUN_DOCK) "cd ~/MCUP \
		&& find ./ -type l -maxdepth 1 |xargs rm -f \
		&& rm -rf dist \
		&& python3 setup.py sdist"

.PHONY: module_local
module_local:
	rm -rf ./dist 
	python3.7 setup.py sdist 

.PHONY: module_test_local
module_local:
	rm -rf ./dist 
	python3.7 setup.py sdist 
	twine check dist/$(MODULE_NAME)-$(MODULE_TEST_VERSION)*

.PHONY: upload
upload:
	$(RUN_DOCK) "twine upload ~/$(MODULE_NAME)/dist/$(MODULE_NAME)-$(MODULE_VERSION)*"

.PHONY: upload_local
upload_local:
	twine upload dist/$(MODULE_NAME)-$(MODULE_VERSION)*

.PHONY: upload_test
upload_test:
	$(RUN_DOCK) "twine upload --repository-url https://test.pypi.org/legacy/ \
		~/$(MODULE_NAME)/dist/$(MODULE_NAME)-$(MODULE_TEST_VERSION)*"

.PHONY: upload_test_local
upload_test_local:
	twine upload --repository-url https://test.pypi.org/legacy/ \
		dist/$(MODULE_NAME)-$(MODULE_TEST_VERSION)*

.PHONY: clean
clean:
	$(RUN_DOCK) "cd ~/MCUP \
		&& rm -rf ./dist ./*.egg-info .tox htmlcov \
		&& find ./ -type l -maxdepth 1 | xargs rm -f \
		&& find ./mcup -type d -name '__pycache__' |xargs rm -rf \
		&& find ./mcup -name '*.pyc' | xargs rm -rf"

.PHONY: clean_local
clean_local:
	rm -rf dist *.egg-info .tox htmlcov
	find ./ -type l -maxdepth 1 | xargs rm -f 
	find ./mcup -type d -name '__pycache__' | xargs rm -rf
	find ./mcup -name '*.pyc' | xargs rm -rf

.PHONY: docs
docs:
	sphinx-apidoc -f -o docs/source mcup tests
	rm docs/source/modules.rst
	python3 setup.py build_sphinx

.PHONY: quick_test
quick_test:
	python3 -m black .
	coverage run -m unittest discover -s ./tests
	coverage html
	sphinx-apidoc -f -o docs/source mcup tests
	rm docs/source/modules.rst
	python3 setup.py build_sphinx