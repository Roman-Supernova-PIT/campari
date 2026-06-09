#!/usr/bin/env bash

pip install -e /home/campari -e /home/snappl
cd /scratch/campari/campari/tests
pip install --upgrade stpsf
export SNPIT_CONFIG=../../examples/perlmutter/campari_config_test.yaml
