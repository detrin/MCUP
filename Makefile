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
MODULE_VERSION = 0.1.7

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
		&& python3 setup.py sdist"

.PHONY: pylint
pylint:
	$(RUN_DOCK) "cd ~/MCUP/mcup \
		&& pylint --rcfile=../.pylintrc * -f parseable"

.PHONY: upload
upload:
	$(RUN_DOCK) "twine upload ~/$(MODULE_NAME)/dist/$(MODULE_NAME)-$(MODULE_VERSION)*"

.PHONY: upload_test
upload_test:
	$(RUN_DOCK) "twine upload --repository-url https://test.pypi.org/legacy/ \
		~/$(MODULE_NAME)/dist/$(MODULE_NAME)-$(MODULE_VERSION)*"


.PHONY: clean
clean:
	$(RUN_DOCK) "cd ~/$(MODULE_NAME) \
		&& rm -rf ./build ./dist ./*.egg-info \
		&& find ./ -type l -maxdepth 1 |xargs rm -f \
		&& find ./$(MODULE) -type d -name '__pycache__' |xargs rm -rf"
