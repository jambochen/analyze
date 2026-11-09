.PHONY: check-report-captions

PYTHON ?= .venv/bin/python

check-report-captions:
	$(PYTHON) scripts/check_report_captions.py

.PHONY: lint
lint:
	$(PYTHON) -m ruff check scripts
	$(PYTHON) scripts/check_conventions.py

.PHONY: check-slides
check-slides:
	$(PYTHON) scripts/check_slide_overflow.py

.PHONY: baseline-timing
baseline-timing:
	$(PYTHON) scripts/baseline_timing.py

.PHONY: api
api:
	$(PYTHON) scripts/module_api.py
