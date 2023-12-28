#!/bin/bash

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

source .venv/bin/activate
.venv/bin/python3 ./src/components/main.py