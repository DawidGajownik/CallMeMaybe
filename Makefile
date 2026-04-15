.PHONY: install run debug clean lint lint-strict reset

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache

reset:
	rm -rf .venv

lint:
	uv run flake8 --exclude=.venv,llm_sdk,moulinette,__pycache__ .
	uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports

lint-strict:
	uv run flake8 --exclude=.venv,__pycache__ .
	uv run mypy src --strictdebug