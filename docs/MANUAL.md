Stručný manuál k aktuálnímu stavu projektu
Co už projekt umí

Projekt aktuálně umí:

načíst raw OHLCV data z data/raw/*.csv

načíst spočítané POC levely z data/processed/*_poc.csv

spojit POC levely pro všechny tickery

určit pro každý level:

jestli je aktuálně kandidát na long nebo short

jestli už byl po svém vzniku tested

datum prvního testu

vzdálenost od aktuální ceny

ATR vzdálenost

kontext trendu

jednoduché skóre kvality

uložit výsledek do:

data/processed/poc_levels_enriched.csv

Hlavní soubory
src/poc_calculator.py

Počítá základní POC levely a ukládá per-ticker CSV do data/processed/.

src/poc_signals.py

Nová vrstva nad hotovými POC levely.
Bere:

data/processed/*_poc.csv

data/raw/*.csv

A vytváří:

data/processed/poc_levels_enriched.csv

src/poc_dashboard.py

Dashboard pro vizualizaci výsledků.
Příště ho napojíme na enriched CSV.

src/poc_backtest.py

Zatím není hlavní priorita. Použijeme až po ověření dashboardu a validity levelů.

Jak projekt teď pustit
1. Přepnout se do projektu
cd ~/Documents/projekty/poc
2. Spustit enrichment logiku
python3 src/poc_signals.py
3. Výstup vznikne zde
data/processed/poc_levels_enriched.csv
Co obsahuje poc_levels_enriched.csv

Důležité sloupce:

Ticker — ticker instrumentu

PeriodType — typ levelu (weekly, monthly, případně další)

Period — konkrétní období

POC / LevelPrice — cena levelu

LevelSide — long nebo short podle aktuální polohy ceny

IsTested — zda byl level po svém vzniku už otestován

FirstTestDate — datum prvního testu

ValidNow — zda je level stále kandidát

DistanceToLastClose — absolutní vzdálenost od aktuální ceny

DistancePct — procentní vzdálenost

DistanceATR — vzdálenost v ATR

TrendContext — up, down, neutral

TrendAligned — zda je level ve směru trendu

Score — jednoduché skóre kvality levelu

Logika tested / untested

Level je považován za testovaný, pokud po svém vzniku:

cena přijde k levelu v definovaném ATR bufferu

a následně udělá reakci dostatečné velikosti

Použité parametry v aktuální verzi:

TOUCH_BUFFER_ATR = 0.15

REACTION_ATR = 0.50

LOOKAHEAD_BARS = 3

To je první mechanická verze logiky, kterou budeme dál ladit.

Aktuální stav

Funguje:

načtení levelů

načtení raw OHLCV

enrichment

export do jednotného CSV

Hotovo bylo úspěšně pro tickery:

DIA

GLD

SPY

TLT

XLE

Celkem:

60 enriched levelů

Co zatím není hotové

dashboard ještě nečte poc_levels_enriched.csv

backtest ještě není napojený na enriched logiku

scoring je zatím jednoduchý

validita levelů je první verze a bude se ještě ladit


### 2) docs/MANUAL.md

```bash
cat > docs/MANUAL.md <<'EOF'
# Manuál projektu

## Přehled

Projekt pracuje s Point of Control (POC) levely vypočítanými z lokálních OHLCV dat a nad nimi staví dvě vrstvy:

1. **signálovou / analytickou vrstvu**
2. **backtest vrstvu**

## Vstupní data

Raw OHLCV data jsou uložena v:

```text
data/raw/
