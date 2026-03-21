# Manual

## Účel dokumentu

Tento manuál popisuje aktuální pracovní stav projektu `poc` a základní způsob používání.

Není to finální dokumentace produkčního systému.
Je to provozní přehled toho, co projekt umí právě teď a jak s ním pracovat při dalším vývoji.

---

## Co projekt řeší

Projekt slouží k vývoji a ověřování obchodních levelů nad historickými daty.

Aktuálně se řeší zejména:
- výpočet **POC (Point of Control)** levelů
- historický backtest návratů ceny k těmto levelům
- vizuální kontrola výsledků
- rozšíření systému o **IB (Initial Balance)** levely
- příprava na budoucí paper trading přes **IB Gateway / TWS**

---

## Pracovní princip

Projekt je stavěný ve vrstvách:

1. **vstupní historická data**
2. **výpočet levelů**
3. **historický backtest**
4. **vizualizace a kontrola**
5. **budoucí paper trading integrace**

To znamená, že nejdřív se musí potvrdit logika levelů a backtestu.
Teprve potom má smysl napojení na živější obchodní infrastrukturu.

---

## Vstupní data

Projekt používá primárně lokální data.

Typicky:
- OHLCV CSV soubory ve `data/raw/`
- pomocné zpracované výstupy ve `data/processed/`

Projekt je laděný tak, aby:
- šel bezpečně testovat offline
- nevyžadoval neustálé online připojení
- používal stejné historické podklady pro opakované backtesty

---

## Hlavní skripty

### `src/poc_calculator.py`
Počítá POC levely z historických dat.

### `src/poc_signals.py`
Pomocná vrstva pro enrichment a další zpracování levelů.

### `src/poc_dashboard.py`
Dashboard pro základní kontrolu levelů a dat.

### `src/poc_backtest.py`
Hlavní backtest logiky návratu ceny k levelům.

### `src/poc_backtest_dashboard.py`
Dashboard pro analýzu výsledků backtestu.

### `src/ib_calculator.py`
Samostatný výpočet IB levelů.
IB se má používat:
- samostatně
- nebo v konfluenci s POC

---

## Aktuální backtest zaměření

Backtest neřeší jen “dotyk levelu”, ale snaží se přiblížit realistickému obchodnímu workflow.

Logika postupně zahrnuje:
- aktivaci levelu až po odchodu ceny od levelu
- validní návrat / clean touch
- invalidaci po nevhodném průrazu nebo gap-cross chování
- omezení na první relevantní touch
- porovnání výsledků podle typu levelu

---

## Režimy, které se mají porovnávat

Cílově má projekt umět porovnat tyto režimy:

### 1. POC-only
Obchodují se pouze POC levely.

### 2. IB-only
Obchodují se pouze IB levely.

### 3. POC + IB confluence
Základní level je POC a IB funguje jako potvrzení nebo filtr.

### 4. IB + POC confluence
Volitelně i opačný režim, pokud bude dávat smysl.

---

## Doporučený způsob práce

Při vývoji držet tento postup:

1. nejdřív zkontrolovat vstupní data
2. potom výpočet levelů
3. potom historický backtest
4. potom dashboard a ruční kontrolu grafů
5. až nakonec paper trading logiku

---

## Základní spuštění

### Výpočet POC
```bash
python3 src/poc_calculator.py
