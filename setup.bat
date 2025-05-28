@echo off
SETLOCAL

echo Creating virtual environment in .venv...
python -m venv .venv

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing the project in editable mode (pip install -e .)...
python -m pip install --upgrade pip
pip install -e .

echo Downloading model if needed...
python setup/download_sam.py

echo Environment setup complete.
echo:
echo To activate the environment later, run:
echo .venv\Scripts\activate