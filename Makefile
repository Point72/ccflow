############################
# Command/Var registration #
############################
PYTHON_VERSION := $(shell python -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor), end='')")
DOCKER := docker
ENV := dev

###############
# Build Tools #
###############
.PHONY: build develop install cubist-sdlc

build:  ## build python
	python -m build .

cubist-sdlc:  ## install prerequisite configuration to be able to install dependencies
	pip install -U --extra-index-url http://artifacts.prod.devops.point72.com/artifactory/api/pypi/dept-ccrt-pypi-published-local/simple --trusted-host artifacts.prod.devops.point72.com cubist-sdlc
	cubist-sdlc configure pypi conda

requirements: cubist-sdlc  ## install python dev dependencies
	python -m pip install toml
	python -m pip install `python -c 'import toml; c = toml.load("pyproject.toml"); print("\n".join(c["build-system"]["requires"]))'`
	python -m pip install `python -c 'import toml; c = toml.load("pyproject.toml"); print("\n".join(c["project"]["optional-dependencies"]["develop"]))'`

develop:  ## install to site-packages in editable mode
	python -m pip install -e .[develop]

install:  ## install to site-packages
	python -m pip install .

###########
# Testing #
###########
.PHONY: test tests coverage

test: ## run the python unit tests
	python -m pytest -v ccflow/tests --junitxml=junit.xml --cov=ccflow --cov-report=xml:.coverage.xml --cov-branch --cov-fail-under=10 --cov-report term-missing

tests: test
coverage: test

###########
# Linting #
###########
.PHONY: lint fix format

lint:  ## lint python with ruff
	python -m ruff check ccflow
	python -m ruff format --check ccflow

lints: lint

fix:  ## autoformat python code with ruff
	python -m ruff check --fix ccflow
	python -m ruff format ccflow

format: fix

#################
# Other Checks #
#################
.PHONY: check checks check-manifest

check: checks

checks: check-manifest  ## run security, packaging, and other checks

check-manifest:  ## run manifest checker for sdist
	check-manifest -v

##############################
# Packaging and Distribution #
##############################
.PHONY: conda dist dist-sdist dist-bdist dist-check publish-conda-dev publish-conda-prod publish-pypi-dev publish-pypi-prod

conda:	## build the conda package
	mkdir -p dist/conda
	conda mambabuild --python $(PYTHON_VERSION) --no-anaconda-upload conda-recipe/ --output-folder ./dist/conda/

setup-conda:  ## install conda build environment
	micromamba create -n cubist-reports-env python=3.9 boa conda-build mamba

dist: dist-sdist dist-bdist dist-check  ## build the pypi/pip installable package

dist-sdist:  ## build sdist
	python -m build -s -n

dist-bdist:  ## build wheel
	python -m build -w -n

dist-check:  ## check the disted assets
	twine check dist/*

publish-conda-dev:  ## publish conda artifact to dev artifactory
	cubist-sdlc artifactory upload conda ${ARTIFACTORY_PASSWORD} --user ${ARTIFACTORY_USERNAME} --env builds

publish-conda-prod:  ## publish conda artifact to prod artifactory
	cubist-sdlc artifactory upload conda ${ARTIFACTORY_PASSWORD} --user ${ARTIFACTORY_USERNAME} --env published

publish-pypi-dev:  ## publish pypi artifact to dev artifactory
	cubist-sdlc artifactory upload pypi ${ARTIFACTORY_PASSWORD} --user ${ARTIFACTORY_USERNAME} --env builds

publish-pypi-prod:  ## publish pypi artifact to prod artifactory
	cubist-sdlc artifactory upload pypi ${ARTIFACTORY_PASSWORD} --user ${ARTIFACTORY_USERNAME} --env published

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


