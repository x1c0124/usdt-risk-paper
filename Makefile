# Makefile for USDT risk paper
.PHONY: all analysis events tables clean-tables fetch

all: analysis events clean-tables tables

analysis:
	python3 run_analysis.py

events:
	python3 scripts/event_study.py

clean-tables:
	python3 scripts/clean_output_tables.py

tables:
	python3 scripts/build_thesis_tables.py

# Data fetching (only run when refreshing). Needs FRED_API_KEY in the environment.
fetch:
	python3 scripts/fetch_usdt_price.py
	python3 scripts/fetch_btc_price.py
	python3 scripts/fetch_macro.py
	python3 scripts/fetch_macro_extended.py
