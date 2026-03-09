# POC Project

Projekt pro výpočet a vyhodnocení Point of Control (POC) levelů nad lokálními OHLCV daty.

## Co projekt aktuálně dělá

- načítá raw OHLCV data z `data/raw/*.csv`
- počítá POC levely po tickeru
- ukládá POC levely do `data/processed/*_poc.csv`
- obohacuje POC levely o obchodní logiku:
  - `long` / `short`
  - `tested` / `untested`
  - `FirstTestDate`
  - `DistanceATR`
  - `TrendContext`
  - `Score`
- ukládá jednotný výstup do:
  - `data/processed/poc_levels_enriched.csv`

## Struktura projektu

```text
poc/
├── data/
│   ├── raw/
│   │   ├── DIA.csv
│   │   ├── GLD.csv
│   │   ├── SPY.csv
│   │   └── ...
│   └── processed/
│       ├── DIA_poc.csv
│       ├── GLD_poc.csv
│       ├── SPY_poc.csv
│       ├── poc_nearest_summary.csv
│       └── poc_levels_enriched.csv
├── input/
│   └── watchlists/
│       └── watchlist.csv
└── src/
    ├── main.py
    ├── poc_calculator.py
    ├── poc_signals.py
    ├── poc_dashboard.py
    ├── poc_backtest.py
    ├── data_fetcher.py
    ├── symbol_loader.py
    ├── utils.py
    └── config.py
Hlavní soubory
src/poc_calculator.py

Počítá základní POC levely a ukládá je do data/processed/*_poc.csv.

src/poc_signals.py

Načítá:

data/processed/*_poc.csv

data/raw/*.csv

A vytváří:

data/processed/poc_levels_enriched.csv

src/poc_dashboard.py

Dashboard pro vizualizaci výsledků.

src/poc_backtest.py

Backtest vrstva. Zatím není hlavní priorita.

Jak spustit enrichment logiku

Z kořene projektu:

python3 src/poc_signals.py
Výstup

Výsledný soubor:

data/processed/poc_levels_enriched.csv
Co obsahuje enriched CSV

Důležité sloupce:

Ticker

PeriodType

Period

POC

LevelPrice

LevelSide

IsTested

FirstTestDate

ValidNow

DistanceToLastClose

DistancePct

DistanceATR

TrendContext

TrendAligned

Score

Aktuální stav

Funguje:

načtení per-ticker POC CSV

načtení raw OHLCV dat

enrichment POC levelů

export do jednotného CSV

Další krok:

napojit dashboard na poc_levels_enriched.csv

vizuálně ověřit validitu levelů

teprve potom řešit backtest


---

## `NEXT_STEPS.md`

```md
# Next Steps

## Cíl další session

Napojit `poc_dashboard.py` na nový soubor:

```text
data/processed/poc_levels_enriched.csv

a začít používat enriched POC levely místo původního jednoduchého přehledu.

Konkrétní kroky

Upravit poc_dashboard.py, aby četl poc_levels_enriched.csv.

Přidat do dashboardu tyto sloupce:

Ticker

PeriodType

Period

POC

LevelSide

IsTested

ValidNow

DistanceATR

TrendContext

Score

Přidat filtry:

pouze ValidNow == True

weekly / monthly

long / short

Přidat řazení:

podle Score

nebo podle absolutní hodnoty DistanceATR

Vizuálně zkontrolovat několik tickerů:

jestli IsTested odpovídá grafu

jestli ValidNow odpovídá realitě

jestli DistanceATR dává smysl

Teprve potom začít řešit poc_backtest.py.

Priorita

Nejdřív:

dashboard

vizuální kontrola

potvrzení logiky

Až potom:

backtest

Poznámky

Aktuální logika tested / untested je první mechanická verze a může se dál ladit.

Použité parametry:

TOUCH_BUFFER_ATR = 0.15

REACTION_ATR = 0.50

LOOKAHEAD_BARS = 3

Stav před další session

Hotovo:

poc_signals.py funguje

vzniká poc_levels_enriched.csv

zpracováno 60 POC levelů pro:

DIA

GLD

SPY

TLT

XLE


---

## `MANUAL.md`

```md
# Manual

## Přehled

Projekt pracuje s Point of Control (POC) levely vypočítanými z lokálních OHLCV dat a následně je obohacuje o základní obchodní logiku.

## Vstupní data

### Raw OHLCV data
Uložena v:

```text
data/raw/

Příklady:

DIA.csv

GLD.csv

SPY.csv

TLT.csv

XLE.csv

POC levely po tickeru

Uloženy v:

data/processed/*_poc.csv

Příklady:

DIA_poc.csv

GLD_poc.csv

SPY_poc.csv

Hlavní workflow
1. Základní POC levely

Skript src/poc_calculator.py vytváří per-ticker POC CSV.

2. Enrichment vrstva

Skript src/poc_signals.py:

načte data/processed/*_poc.csv

načte data/raw/*.csv

spočítá ATR a EMA kontext

určí:

LevelSide

IsTested

FirstTestDate

DistanceATR

TrendContext

Score

uloží výstup do:

data/processed/poc_levels_enriched.csv

Jak spustit

Z kořene projektu:

python3 src/poc_signals.py
Výstupní soubor
data/processed/poc_levels_enriched.csv
Význam důležitých sloupců
Ticker

Ticker instrumentu.

PeriodType

Typ levelu, například:

weekly

monthly

Period

Konkrétní období, ze kterého level vznikl.

POC / LevelPrice

Cena Point of Control levelu.

LevelSide

Interpretace levelu podle aktuální ceny:

long = cena je nad levelem, level je kandidát na support

short = cena je pod levelem, level je kandidát na resistance

IsTested

Jestli už byl level po svém vzniku otestován.

FirstTestDate

Datum prvního testu levelu.

ValidNow

Jestli je level stále kandidát pro další sledování.

DistanceToLastClose

Absolutní vzdálenost mezi posledním close a levelem.

DistancePct

Procentní vzdálenost mezi posledním close a levelem.

DistanceATR

Vzdálenost od levelu vyjádřená v ATR.

TrendContext

Jednoduché určení trendu:

up

down

neutral

TrendAligned

Jestli je level ve směru trendu.

Score

Jednoduché skóre kvality levelu.

Logika tested / untested

Level je považován za testovaný, pokud po svém vzniku:

cena přijde k levelu v definovaném ATR bufferu

následně udělá reakci dostatečné velikosti

Aktuální parametry:

TOUCH_BUFFER_ATR = 0.15

REACTION_ATR = 0.50

LOOKAHEAD_BARS = 3

To je první mechanická verze logiky, která bude dál laděna.

Aktuální stav projektu

Funguje:

načtení raw OHLCV

načtení POC levelů

obohacení levelů o validitu a vzdálenost

export do jednotného CSV

Zpracované tickery:

DIA

GLD

SPY

TLT

XLE

Celkem:

60 enriched levelů

Co zatím není hotové

dashboard ještě nečte enriched CSV

scoring je zatím základní

backtest ještě není napojený na enriched logiku

parametry tested/untested se budou dál ladit

Doporučení

Další krok je napojit dashboard na enriched CSV a vizuálně ověřit, že logika validních a testovaných levelů odpovídá realitě v grafu.


---

Ještě mě napadá přidat i jednoduchý `.gitignore`, aby se do repa netlačily zbytečnosti. Třeba takto:

## `.gitignore`

```gitignore
__pycache__/
*.pyc
*.pyo
*.pyd
.venv/
venv/
.env
.DS_Store
