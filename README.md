# POC Project

Python projekt pro práci s Point of Control (POC) levely.

## Cíl projektu

Projekt bude postupně umět:

- načítat historická data
- stahovat data z Interactive Brokers
- počítat POC levely
- sledovat validitu levelů
- exportovat výsledky do CSV

## Aktuální stav

První základní kostra projektu.

## Spuštění

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
