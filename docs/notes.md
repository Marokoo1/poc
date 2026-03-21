# Notes

## Projektový kontext

Projekt `poc` je pracovní vývojový projekt pro návrh a testování obchodní logiky nad historickými levely.

Hlavní osa projektu:
- POC levely
- IB levely
- jejich samostatné i společné testování
- následný přenos ověřené logiky do paper tradingu přes IB Gateway

---

## Důležitá rozhodnutí

### 1. Nejprve lokální data, potom broker napojení
Vývoj a ladění probíhá nejdřív nad lokálními CSV / raw daty.

Důvod:
- opakovatelnost testů
- rychlejší ladění
- menší chaos při vývoji
- oddělení logiky levelů od exekuční vrstvy

### 2. POC a IB počítat odděleně
POC a IB nemají být slepené do jedné nejasné vrstvy už při výpočtu.

Správný přístup:
- POC počítat samostatně
- IB počítat samostatně
- spojovat je až v backtestu a v obchodní logice

### 3. Musí existovat tři samostatné testovací režimy
- POC-only
- IB-only
- POC+IB confluence

To je důležité pro férové porovnání výsledků.

### 4. Confluence není nový level
Confluence je vlastnost nebo filtr obchodu, ne samostatný “slepený” level.

### 5. Paper trading až po ověření backtestu
Napojení na TWS / IB Gateway přijde až tehdy, když bude logika systému skutečně dává smysl historicky i vizuálně.

---

## Praktické zásady pro další vývoj

- nerozbíjet fungující POC workflow
- nové vrstvy přidávat odděleně a čitelně
- nepsat dokumentaci dopředu pro věci, které ještě nejsou hotové
- důležité změny nejdřív ověřit graficky
- dashboard používat jako kontrolní nástroj, ne jen jako “hezký výstup”

---

## Aktuální technický směr

Krátkodobě:
- dokončit IB výpočty a jejich zapojení do backtestu
- porovnat výsledky proti samotnému POC

Střednědobě:
- vytvořit logiku paper trading workflow
- napojit systém na IB Gateway / TWS
- testovat paper execution

Dlouhodobě:
- proměnit ověřenou logiku v obchodovatelný systematický workflow

---

## Poznámka

Tento soubor má zachycovat rozhodnutí a směr.
Ne detailní úkoly.
Na detailní úkoly slouží `todo.md` a `next_steps.md`.
