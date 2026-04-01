.PHONY: install run debug clean lint lint-strict

install:
	uv venv
	uv pip install -r llm_sdk/pyproject.toml

run:
	uv run python -m main

debug:
	python3 -m pdb main.py

clean:
	rm -rf __pycache__ .mypy_cache llm_sdk/__pycache__ llm_sdk/llm_sdk/__pycache__

lint:
	flake8 . && mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 . && mypy . --strict
