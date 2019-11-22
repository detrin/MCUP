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

ifeq ($(user),)
# USER retrieved from env, UID from shell.
HOST_USER ?= $(strip $(if $(USER),$(USER),nodummy))
HOST_UID ?= $(strip $(if $(shell id -u),$(shell id -u),4000))
else
# allow override by adding user= and/ or uid=  (lowercase!).
# uid= defaults to 0 if user= set (i.e. root).
HOST_USER = $(user)
HOST_UID = $(strip $(if $(uid),$(uid),0))
endif

# cli prefix for commands to run in container
RUN_DOCK = \
	docker-compose -p $(MODULE_NAME)_$(HOST_UID) run --rm $(SERVICE_TARGET) sh -l -c

# export such that its passed to shell functions for Docker to pick up.
export MODULE_NAME
export HOST_USER
export HOST_UID



.PHONY: shell 
shell:
	$(RUN_DOCK) "cd ~/$(MODULE_NAME) \
		&& pip install -r requirements.txt \
		&& bash"

.PHONY: module
module: 
	@# ensure there is a symlink from MODULE_NAME to module directory
	@# then run regular setup.py to build the module
	$(RUN_DOCK) "cd ~/$(MODULE_NAME) \
		&& find ./ -type l -maxdepth 1 |xargs rm -f \
		&& python3 setup.py sdist"

.PHONY: pylint
pylint:
	$(RUN_DOCK) "cd ~/$(MODULE_NAME)/$(MODULE_NAME) \
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
