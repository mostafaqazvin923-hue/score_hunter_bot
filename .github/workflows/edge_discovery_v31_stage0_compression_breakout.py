name: Run V31 Stage-0 Compression Breakout Research
on:
  workflow_dispatch:
jobs:
  research:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pandas numpy requests
      - name: Compile check
        run: python -m py_compile edge_discovery_v31_stage0_compression_breakout.py
      - name: Run V31 Stage-0
        run: python edge_discovery_v31_stage0_compression_breakout.py
      - name: Upload research artifacts
        uses: actions/upload-artifact@v4
        with:
          name: v31-stage0-compression-breakout
          path: reports/xt_v31_stage0_compression_breakout/
          if-no-files-found: error
