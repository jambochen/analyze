.PHONY: check-report-captions

PYTHON ?= .venv/bin/python

check-report-captions:
	$(PYTHON) scripts/check_report_captions.py

.PHONY: lint
lint:
	$(PYTHON) -m ruff check scripts

.PHONY: check-slides
check-slides:
	$(PYTHON) scripts/check_slide_overflow.py
