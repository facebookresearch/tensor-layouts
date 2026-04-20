.PHONY: build clean test check docs lint examples

build:
	pip install -e .

clean:
	rm -rf build/ dist/ src/*.egg-info
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf examples_output/ tests/figures/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.py[cod]' -delete

test:
	python -m pytest tests/ -v

check: test

docs:
	python3 docs/generate_figures.py

lint:
	ruff check src/ tests/ examples/

examples:
	python3 examples/layouts.py
	python3 examples/tensor.py
	python3 examples/viz.py
