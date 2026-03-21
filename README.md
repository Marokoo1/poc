# POC Project

Projekt pro výpočet, vizualizaci a backtest obchodních levelů nad lokálními OHLCV daty.

Aktuální zaměření projektu:
- historický výpočet a testování **POC (Point of Control)** levelů
- rozšíření backtestu o **IB (Initial Balance)** levely
- porovnání režimů:
  - **POC**
  - **IB**
  - **POC + IB confluence**
- příprava podkladů pro další krok:
  - **paper trading přes IB Gateway / TWS**

---

## Hlavní cíl projektu

Cílem není jen vypočítat levely, ale ověřit, zda jsou obchodovatelné v systematické podobě.

Projekt má odpovědět hlavně na tyto otázky:
- fungují samotné **POC** levely?
- fungují samotné **IB** levely?
- přináší **konfluence POC + IB** lepší výsledky než samotné POC?
- jaká logika vstupu, invalidace a řízení obchodu je realisticky obchodovatelná?

---

## Co projekt aktuálně řeší

### 1. Výpočet levelů
Projekt pracuje nad lokálními OHLCV daty uloženými v projektu a počítá:
- POC levely po definovaných periodách
- historické levely pro backtest
- pomocné výstupy pro dashboard a kontrolu logiky

### 2. Historický backtest
Backtest vrstva testuje návraty ceny k levelům a obsahuje postupně laděnou logiku:
- aktivace levelu až po odchodu ceny od levelu
- clean touch / valid touch logiku
- invalidaci po nevhodném průrazu nebo gap-cross situaci
- omezení na první validní test
- filtrování a porovnávání výsledků podle typu levelu

### 3. Vizualizace a kontrola
Projekt obsahuje dashboardy pro:
- kontrolu levelů v grafu
- procházení historických obchodů
- filtrování výsledků backtestu
- porovnávání period, směrů a typů levelů

### 4. Rozpracované rozšíření o IB
Další vývojová větev přidává:
- roční IB
- měsíční IB
- standardní projekce
- volitelné fib projekce
- test režimů:
  - POC-only
  - IB-only
  - POC+IB confluence

---

## Aktuální stav projektu

Projekt je ve fázi:
1. lokální výpočet a backtest levelů z CSV / raw dat
2. průběžné ladění logiky levelů a vstupů
3. rozšíření o IB
4. příprava na další etapu:
   **napojení paper tradingu na IB Gateway**

Je důležité:
- nejdřív dokončit a ověřit historický backtest
- teprve potom přenášet logiku do paper trading workflow

---

## Struktura projektu

```text
poc/
├── config/
├── docs/
│   ├── MANUAL.md
│   ├── next_steps.md
│   ├── notes.md
│   ├── plan.md
│   ├── tahak
│   └── todo.md
├── input/
│   └── watchlists/
├── poc/
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
