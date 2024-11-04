###############
# Build Tools #
###############
.PHONY: build develop install

build:  ## build python/javascript
	python -m build .

requirements:  ## install prerequisite python build requirements
	python -m pip install --upgrade pip toml
	python -m pip install `python -c 'import toml; c = toml.load("pyproject.toml"); print("\n".join(c["build-system"]["requires"]))'`
	python -m pip install `python -c 'import toml; c = toml.load("pyproject.toml"); print(" ".join(c["project"]["optional-dependencies"]["develop"]))'`

develop:  ## install to site-packages in editable mode
	python -m pip install -e .[develop]

install:  ## install to site-packages
	python -m pip install .

###########
# Testing #
###########
.PHONY: test tests

test: ## run the python unit tests
	python -m pytest -v ccflow/tests --cov=ccflow --cov-report xml --cov-report term-missing

test: tests

###########
# Linting #
###########
.PHONY: lint fix format

lint-py:  ## lint python with ruff
	python -m ruff check ccflow setup.py
	python -m ruff format --check ccflow setup.py

lint-docs:  ## lint docs with mdformat and codespell
	python -m mdformat --check docs/wiki/ README.md
	python -m codespell_lib docs/wiki/ README.md

fix-py:  ## autoformat python code with ruff
	python -m ruff check --fix ccflow setup.py
	python -m ruff format ccflow setup.py

fix-docs:  ## autoformat docs with mdformat and codespell
	python -m mdformat docs/wiki/ README.md
	python -m codespell_lib --write docs/wiki/ README.md

lint: lint-py lint-docs  ## run all linters
lints: lint
fix: fix-py fix-docs  ## run all autoformatters
format: fix

#################
# Other Checks #
#################
.PHONY: check checks check-manifest

check: checks

checks: check-manifest  ## run security, packaging, and other checks

check-manifest:  ## run manifest checker for sdist
	check-manifest -v

################
# Distribution #
################
.PHONY: dist publish

dist: clean build  ## create dists
	python -m twine check dist/*

publish: dist  ## dist to pypi
	python -m twine upload dist/* --skip-existing

############
# Cleaning #
############
.PHONY: clean

clean: ## clean the repository
	find . -name "__pycache__" | xargs  rm -rf
	find . -name "*.pyc" | xargs rm -rf
	find . -name ".ipynb_checkpoints" | xargs  rm -rf
	rm -rf .coverage coverage *.xml build dist *.egg-info lib node_modules .pytest_cache *.egg-info
	git clean -fd

###########
# Helpers #
###########
.PHONY: help

# Thanks to Francoise at marmelab.com for this
.DEFAULT_GOAL := help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

print-%:
	@echo '$*=$($*)'
