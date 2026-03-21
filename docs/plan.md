# Plan

## Hlavní cíl projektu

Vybudovat workflow pro systematické obchodování levelů, které bude:
1. historicky otestované
2. vizuálně ověřené
3. přenositelné do paper tradingu přes IB Gateway

---

## Fáze projektu

### Fáze 1 — Základní datová a výpočetní vrstva
- práce s lokálními OHLCV daty
- výpočet POC
- ukládání a opakované používání výsledků

### Fáze 2 — Historický backtest POC
- aktivace levelu
- clean touch logika
- invalidace
- první validní touch
- dashboard pro kontrolu výsledků

### Fáze 3 — Rozšíření o IB
- roční IB
- měsíční IB
- standardní projekce
- volitelné fib projekce
- samostatný IB režim

### Fáze 4 — Confluence test
- POC-only
- IB-only
- POC+IB
- vyhodnocení přínosu konfluence

### Fáze 5 — Přenesení systému do paper tradingu
- napojení na TWS / IB Gateway
- čtení obchodních signálů
- logika zadávání a správy obchodů
- paper workflow

### Fáze 6 — Stabilizace a další automatizace
- zjednodušení workflow
- případná denní rutina
- robustnější monitoring a provozní logika

---

## Co musí být jasné před paper tradingem

Než se začne řešit IB Gateway vrstva, musí být jasné:

- jaké levely se budou obchodovat
- jaké filtry jsou povinné
- co je vstupní podmínka
- co je invalidace
- kde je stop-loss
- kde je profit target
- co dělat při gapu nebo nevhodném průrazu
- jaké typy obchodů chceme na paperu skutečně simulovat

---

## Aktuální pracovní priorita

Teď je hlavní priorita:
- dokončit POC + IB historický backtest
- udělat férové srovnání režimů
- rozhodnout, co má smysl převést do paper tradingu

---

## Pracovní zásada

Projekt se má posouvat po vrstvách:

1. výpočet
2. backtest
3. dashboard kontrola
4. paper trading integrace

Nepřeskakovat rovnou na exekuci bez potvrzené logiky.
