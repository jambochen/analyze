.PHONY: check-report-captions

PYTHON ?= .venv/bin/python

check-report-captions:
	$(PYTHON) scripts/check_report_captions.py
