SHELL=/bin/bash

SYSTEM_PYTHON=python3
VENV:=venv

PIP:=$(VENV)/bin/pip
PYTHON:=$(VENV)/bin/python3

$(VENV):
	$(SYSTEM_PYTHON) -m venv venv

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
	$(PYTHON) -m mkdocs build

.PHONY: mkdocs-serve
mkdocs-serve: pip-install
	$(PYTHON) -m mkdocs serve

