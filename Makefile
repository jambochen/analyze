.PHONY: report-pdfs check-report-captions

PYTHON ?= .venv/bin/python

check-report-captions:
	$(PYTHON) scripts/check_report_captions.py

report-pdfs: check-report-captions
	typst compile --root . report/proposal_doc.typ report/proposal_doc.pdf
	typst compile --root . report/proposal.typ report/proposal.pdf
	typst compile --root . report/methods_doc.typ report/methods_doc.pdf
	typst compile --root . report/methods.typ report/methods.pdf
