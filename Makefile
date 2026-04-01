.PHONY: install run debug clean lint lint-strict

install:
	uv venv
	uv pip install -r llm_sdk/pyproject.toml
	uv pip install transformers torch accelerate

run:
	uv run python -m main

debug:
	uv run python -m pdb main.py

clean:
	rm -rf __pycache__ .mypy_cache llm_sdk/__pycache__ llm_sdk/llm_sdk/__pycache__ .venv

lint:
	uv run flake8 . && uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 . && uv run mypy . --strict