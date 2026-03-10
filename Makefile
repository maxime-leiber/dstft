.PHONY: format lint test docs

format:
	ruff format src tests
	ruff check --fix src tests

lint:
	ruff format --check src tests
	ruff check src tests

test:
	pytest

docs:
	$(MAKE) -C docs html
