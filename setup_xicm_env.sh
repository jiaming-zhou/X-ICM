#!/usr/bin/env bash
set -e

echo "Installing X-ICM dependencies..."

pip install -r requirements_xicm_5090.txt

echo "Installing OpenAI CLIP..."
pip install git+https://github.com/openai/CLIP.git

echo "Installing local packages..."
pip install -e ./RLBench --no-deps
pip install -e ./PyRep --no-deps
pip install -e ./YARR --no-deps

echo "Done."
echo "Remember to export CoppeliaSim environment variables before running eval."