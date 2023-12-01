#!/usr/bin/env sh

. .venv/bin/activate
python -m mypy ./src/rtctools_heat_network ./tests/
