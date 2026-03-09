Plan na příště
Cíl další session

!!! DASBORD se NESPOUSTI SAM PO CALCULATOR.py !!!

Napojit poc_dashboard.py na nový soubor data/processed/poc_levels_enriched.csv a začít používat enriched POC levely místo původního jednoduchého přehledu.

Konkrétní kroky

Upravit poc_dashboard.py, aby četl poc_levels_enriched.csv.

Přidat do dashboardu nové sloupce:

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

Seřazení:

podle Score

nebo podle absolutní hodnoty DistanceATR

Vizuálně zkontrolovat několik tickerů:

jestli IsTested dává smysl proti grafu

jestli ValidNow odpovídá realitě

Teprve potom začít řešit poc_backtest.py.

Priorita

Nejdřív vizualizace a kontrola logiky.
Backtest až po potvrzení, že enriched levely sedí.
