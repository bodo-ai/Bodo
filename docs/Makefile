SHELL=/bin/bash

# Assume Pixi handles environment, use system Python for any fallback
PYTHON:=pixi run python
PIP:=pixi run pip

# Check if PLATFORM_SDK_PATH is set, fail early
ifndef PLATFORM_SDK_PATH
$(error PLATFORM_SDK_PATH is not set)
endif

#
# Dependencies
#
.PHONY: pip-install
pip-install:
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

#
# MIKE
#
.PHONY: mike-deploy
mike-deploy: pip-install
	PYTHONPATH=$(PLATFORM_SDK_PATH) pixi run mike deploy $(version_name)

.PHONY: mike-serve
mike-serve: pip-install
	PYTHONPATH=$(PLATFORM_SDK_PATH) pixi run mike serve

.PHONY: mike-push
mike-push: pip-install
	PYTHONPATH=$(PLATFORM_SDK_PATH) pixi run mike deploy $(version_name) --push

.PHONY: mike-push-latest
mike-push-latest: pip-install
	PYTHONPATH=$(PLATFORM_SDK_PATH) pixi run mike deploy --update-alias $(version_name) latest --push
