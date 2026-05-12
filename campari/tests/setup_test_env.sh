#!/usr/bin/env bash
set -euo pipefail

pip install -e /scratch/campari -e /scratch/snappl
cd /scratch/campari/campari/tests
pip install --upgrade stpsf
export SNPIT_CONFIG=../../examples/perlmutter/campari_config_test.yaml
