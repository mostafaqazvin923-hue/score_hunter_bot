name: Run Backtest Setup 4 V1

on:
  workflow_dispatch:

jobs:
  backtest:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pandas numpy requests

      - name: Compile check
        run: |
          python -m py_compile backtest_setup4_v1.py

      - name: Run Setup 4 V1
        run: |
          python backtest_setup4_v1.py

      - name: Upload backtest results
        uses: actions/upload-artifact@v4
        with:
          name: setup4-v1-results
          path: setup4_v1_outputs/
