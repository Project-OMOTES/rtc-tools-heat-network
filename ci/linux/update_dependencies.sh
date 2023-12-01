#!/usr/bin/env sh

. .venv/bin/activate
pip-compile -U --output-file=requirements.txt pyproject.toml
pip-compile -U --extra=dev --output-file=dev-requirements.txt pyproject.toml
