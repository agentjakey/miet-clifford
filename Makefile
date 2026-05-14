# Makefile for miet-clifford
# Run all targets from the repo root.
#
# The full 2-3 hour sweep (run_sweep.py) is intentionally NOT a default target.
# To regenerate simulation data: python miet/scripts/run_sweep.py --seed 42
#
# On Windows with MiKTeX, set PDFLATEX and BIBTEX explicitly:
#   make report PDFLATEX="C:/Users/.../MiKTeX/miktex/bin/x64/pdflatex.exe" \
#               BIBTEX="C:/Users/.../MiKTeX/miktex/bin/x64/bibtex.exe"

PYTHON      ?= python
PDFLATEX    ?= pdflatex
BIBTEX      ?= bibtex
MIET        := miet
REPORT_DIR  := $(MIET)/report

.PHONY: install test audit quick analyze figures report clean-generated help

help:
	@echo "Targets:"
	@echo "  install          pip install -e .[dev]"
	@echo "  test             pytest (53 tests)"
	@echo "  audit            physics correctness audit (7 checks, ~30s)"
	@echo "  quick            quick simulation sweep, L in {8,12} (~30s)"
	@echo "  analyze          crossing + FSS + log-scaling + summary tables"
	@echo "  figures          all figures (runs analyze, then circuit schematic)"
	@echo "  report           compile LaTeX report (requires pdflatex + bibtex)"
	@echo "  clean-generated  remove LaTeX auxiliary files"
	@echo ""
	@echo "Typical workflow after cloning:"
	@echo "  make install"
	@echo "  make test"
	@echo "  make audit"
	@echo "  make quick        # or run_sweep.py for the full 2-3h sweep"
	@echo "  make figures"
	@echo "  make report"

install:
	pip install -e ".[dev]"

test:
	pytest

audit:
	$(PYTHON) $(MIET)/scripts/physics_audit.py

# quick must run from miet/ because SAVE_PATH = "data/..." is CWD-relative
quick:
	cd $(MIET) && $(PYTHON) scripts/run_quick.py --seed 42

# analyze runs the four analysis scripts in dependency order:
#   phase_diagram  -> writes crossing_table.txt / crossing_table.json
#   finite_size    -> writes fss_config.json / fss_sensitivity.txt
#   log_scaling    -> reads fss_config.json and crossing_table.txt
#   critical_fit   -> reads all data, writes critical_quantities.txt
analyze:
	$(PYTHON) $(MIET)/analysis/phase_diagram.py
	$(PYTHON) $(MIET)/analysis/finite_size.py
	$(PYTHON) $(MIET)/analysis/log_scaling.py
	$(PYTHON) $(MIET)/analysis/critical_fit.py

# figures runs analyze (which produces figs 1-3) then the schematic (fig 0)
figures: analyze
	$(PYTHON) $(MIET)/analysis/circuit_schematic.py

# Four-pass LaTeX compile; copies output PDF to repo root
report:
	cd $(REPORT_DIR) && \
	$(PDFLATEX) -interaction=nonstopmode main.tex && \
	$(BIBTEX) main && \
	$(PDFLATEX) -interaction=nonstopmode main.tex && \
	$(PDFLATEX) -interaction=nonstopmode main.tex
	cp $(REPORT_DIR)/main.pdf miet_research_report.pdf
	@echo "Report written to miet_research_report.pdf"

clean-generated:
	cd $(REPORT_DIR) && rm -f \
		*.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk
	@echo "LaTeX auxiliary files removed (PDF and source preserved)"
