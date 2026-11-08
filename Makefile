PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; elif [ -x /opt/anaconda3/bin/python ]; then echo /opt/anaconda3/bin/python; elif command -v python3 >/dev/null 2>&1; then command -v python3; else command -v python; fi)
export PYTHONPATH := $(CURDIR)/src$(if $(PYTHONPATH),:$(PYTHONPATH))

.PHONY: install test data features replicate extend all

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

data:
	$(PYTHON) scripts/01_get_data.py

features:
	$(PYTHON) scripts/02_build_dataset.py

replicate:
	$(PYTHON) scripts/03_replicate.py

extend:
	$(PYTHON) scripts/04_extend.py

all: data features replicate test
