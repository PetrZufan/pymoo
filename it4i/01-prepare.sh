#!/bin/bash

ml Python/3.9.5-GCCcore-10.3.0
pip install tensorflow --user
pip install tensorflow_io --user
pip install autograd --user
pip install scipy --user
pip install matplotlib --user

# TODO: run in repo root directory (eg ~/fit/git/pymoo/)
# make compile
pip install ../ --user
