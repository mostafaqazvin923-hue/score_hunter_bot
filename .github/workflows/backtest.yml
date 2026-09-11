name: Run Backtest V5 CLEAN

on:
  workflow_dispatch:

jobs:
  run-backtest:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout Code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install ccxt pandas numpy

      - name: Check Backtest File
        run: |
          echo "Files in repository:"
          ls -la
          echo "Checking backtest_v5.py:"
          test -f backtest_v5.py

      - name: Run Backtest V5 CLEAN
        run: |
          python backtest_v5.py
