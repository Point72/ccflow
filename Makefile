###############
# Build Tools #
###############
.PHONY: build develop install

build:  ## build python
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
	python -m pytest -v ccflow/tests --junitxml=junit.xml --cov=ccflow --cov-report=xml:.coverage.xml --cov-branch --cov-fail-under=10 --cov-report term-missing

tests: test

###########
# Linting #
###########
.PHONY: lint fix format

lint:  ## lint python with isort and ruff
	python -m isort ccflow setup.py --check
	python -m ruff check ccflow setup.py
	python -m ruff format --check ccflow setup.py

lints: lint

fix:  ## autoformat python code with isort and ruff
	python -m isort ccflow setup.py
	python -m ruff format ccflow setup.py

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
.PHONY: dist dist-check publish

dist: clean build dist-check  ## create dists
	python -m twine check dist/*

dist-check:  ## check the dists
	python -m twine check dist/*

publish:  ## dist to pypi
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

