# Makefile - Eagle Real-Time Semantic Surveillance
# Common developer commands for testing, linting, docker, and demos.
# Usage: make <target>   |   make help : for full list

SHELL := /bin/bash

.PHONY: install install-frontend setup test lint coverage up down demo clean help

.DEFAULT_GOAL := help

# --------------------------------------------------------------------------
# install : Install backend dependencies
# --------------------------------------------------------------------------
install:
	@echo "Installing backend dependencies..."
	pip install -r services/detection/requirements.txt
	pip install -r requirements-dev.txt
	@echo "Done: backend dependencies installed."

# --------------------------------------------------------------------------
# install-frontend : Install frontend npm dependencies
# --------------------------------------------------------------------------
install-frontend:
	@echo "Installing frontend dependencies..."
	cd apps/frontend && npm install
	@echo "Done: frontend dependencies installed."

# --------------------------------------------------------------------------
# setup : Full developer environment (backend + frontend)
# --------------------------------------------------------------------------
setup: install install-frontend
	@echo "Done: full development environment ready."

# --------------------------------------------------------------------------
# test : Run the full test suite with pytest
# --------------------------------------------------------------------------
test:
	@echo "Running tests..."
	pytest tests/ -v

# --------------------------------------------------------------------------
# lint : Run ruff and black formatting checks
# --------------------------------------------------------------------------
lint:
	@echo "Running linters..."
	python -m ruff check .
	python -m black --check .
	@echo "Done: lint passed."

# --------------------------------------------------------------------------
# coverage : Run tests with coverage scoped to services/
# --------------------------------------------------------------------------
coverage:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=services --cov-report=term-missing

# --------------------------------------------------------------------------
# up : Start all services with docker compose
# --------------------------------------------------------------------------
up:
	@echo "Starting services..."
	docker compose up -d
	@echo "Done: services running."

# --------------------------------------------------------------------------
# down : Stop all docker compose services
# --------------------------------------------------------------------------
down:
	@echo "Stopping services..."
	docker compose down
	@echo "Done: services stopped."

# --------------------------------------------------------------------------
# demo : Run the detection demo
# --------------------------------------------------------------------------
demo:
	@echo "Running detection demo..."
	python services/detection/detection.py

# --------------------------------------------------------------------------
# clean : Remove temporary and cache files
# --------------------------------------------------------------------------
clean:
	@echo "Cleaning up..."
	python -c "import shutil, pathlib; p = pathlib.Path('.');\
	[shutil.rmtree(d, ignore_errors=True) for d in p.rglob('__pycache__')];\
	[shutil.rmtree(d, ignore_errors=True) for d in p.rglob('.pytest_cache')];\
	[shutil.rmtree(d, ignore_errors=True) for d in p.rglob('.ruff_cache')];\
	[shutil.rmtree(d, ignore_errors=True) for d in p.rglob('htmlcov')];\
	[f.unlink(missing_ok=True) for f in p.rglob('*.pyc')];\
	[f.unlink(missing_ok=True) for f in p.rglob('.coverage')]"
	@echo "Done: clean complete."

# --------------------------------------------------------------------------
# help : Print this usage summary
# --------------------------------------------------------------------------
help:
	@echo ""
	@echo "  Eagle - Developer Commands"
	@echo "  ----------------------------------------------------------"
	@echo "  make install          - Install backend dependencies"
	@echo "  make install-frontend - Install frontend npm dependencies"
	@echo "  make setup            - Full dev setup (backend + frontend)"
	@echo "  make test             - Run pytest suite"
	@echo "  make lint             - Run ruff and black checks"
	@echo "  make coverage         - Run tests with coverage (services/)"
	@echo "  make up               - Start docker services"
	@echo "  make down             - Stop docker services"
	@echo "  make demo             - Run detection demo"
	@echo "  make clean            - Remove temporary/cache files"
	@echo "  make help             - Print this usage summary"
	@echo "  ----------------------------------------------------------"
	@echo ""
  