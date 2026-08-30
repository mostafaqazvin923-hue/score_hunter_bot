name: Score Hunter Bot Runner

on:
  schedule:
    - cron: '*/15 * * * *'  # اجرا در هر ۱۵ دقیقه
  workflow_dispatch:        # قابلیت اجرای دستی (دکمه Run workflow)

jobs:
  run-bot:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install requests

      - name: Run Bot
        run: python bot.py
