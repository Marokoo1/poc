cat > README.md <<'EOF'
# POC Project

Projekt pro výpočet, vizualizaci a backtest Point of Control (POC) levelů nad lokálními OHLCV daty.

## Co projekt aktuálně dělá

- načítá raw OHLCV data z `data/raw/*.csv`
- počítá POC levely po tickeru
- ukládá POC levely do `data/processed/*_poc.csv`
- vytváří enriched přehled levelů v `data/processed/poc_levels_enriched.csv`
- umožňuje vizuální kontrolu levelů v dashboardu
- provádí historický backtest POC návratů v `src/poc_backtest.py`
- ukládá backtest výstupy do:
  - `data/processed/poc_backtest_levels.csv`
  - `data/processed/poc_backtest_trades.csv`
  - `data/processed/poc_backtest_summary.csv`

## Hlavní logika backtestu

Aktuální verze backtestu už nepoužívá slepé vstupy hned po uzavření periody.

Nová logika obsahuje:

- **departure filter**  
  level je obchodovatelný až poté, co cena od POC odejde o nastavitelnou vzdálenost

- **clean touch logic**  
  vstup se bere jen při čistém návratu k levelu

- **gap-cross invalidation**  
  pokud cena gapne skrz level, obchod se neotevře opačným směrem

- **rotation filter**  
  pokud se cena kolem POC začne jen točit, level se vyřadí

- **first valid touch only**  
  obchoduje se jen první validní návrat k levelu

## Struktura projektu

```text
poc/
├── config/
├── docs/
├── input/
│   └── watchlists/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── main.py
│   ├── poc_calculator.py
│   ├── poc_signals.py
│   ├── poc_dashboard.py
│   ├── poc_backtest.py
│   ├── poc_backtest_dashboard.py
│   ├── data_fetcher.py
│   ├── symbol_loader.py
│   ├── utils.py
│   └── config.py
├── README.md
└── requirements.txt
