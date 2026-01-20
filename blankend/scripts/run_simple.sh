#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export SIMPLE_FORWARD=1
python3 -m uvicorn blankend.main:app --host 0.0.0.0 --port 8001
