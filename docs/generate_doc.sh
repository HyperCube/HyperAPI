#!/bin/bash

set -e

export PYTHONPATH=../HyperAPI/hyper_api

sphinx-build $1 $2 -c .
