.PHONY: style fixup test build-release

# Ensure we use local version
export PYTHONPATH = src

check_dirs := src tests examples

style:
	ruff check $(check_dirs) setup.py --fix
	ruff format $(check_dirs) setup.py

fixup: style

test:
	pytest -n auto -v tests/

build-release:
	rm -rf dist build
	python -m build
