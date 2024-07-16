SHELL=/bin/bash


SYSTEM_PYTHON=python3
VENV:=venv

PIP:=$(VENV)/bin/pip
PYTHON:=$(VENV)/bin/python3

$(VENV):
	$(SYSTEM_PYTHON) -m venv venv

# check if $(PLATFORM_SDK_PATH) is set, if not then throw error
ifndef PLATFORM_SDK_PATH
$(error PLATFORM_SDK_PATH is not set)
endif



#
# Dependencies
#
.PHONY: pip-install
pip-install: $(VENV)
	$(PYTHON) -m ensurepip --upgrade
	$(PIP) install -r requirements.txt


#
# MKDOCS
#
.PHONY: mkdocs-build
mkdocs-build: pip-install
	PYTHONPATH=$(PLATFORM_SDK_PATH) $(PYTHON) -m mkdocs build

.PHONY: mkdocs-serve
mkdocs-serve: pip-install
	PYTHONPATH=$(PLATFORM_SDK_PATH) $(PYTHON) -m mkdocs serve


.PHONY: mike-deploy
mike-deploy: pip-install
	PYTHONPATH=$(PLATFORM_SDK_PATH) mike deploy $(version_name)

.PHONY: mike-serve
mike-serve: pip-install
	PYTHONPATH=$(PLATFORM_SDK_PATH) mike serve

.PHONY: mike-push
mike-push: pip-install
	PYTHONPATH=$(PLATFORM_SDK_PATH) mike deploy $(version_name) --push

.PHONY: mike-push-latest
mike-push-latest: pip-install
	PYTHONPATH=$(PLATFORM_SDK_PATH) mike deploy --update-alias $(version_name) latest --push

