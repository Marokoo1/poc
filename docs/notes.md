Priste!! přidat TLS parser pro AmiBroker/YahooDownloader workflow

začít konečně psát první skutečný výpočet měsíčního POC

Za mě je lepší nejdřív TLS parser, pokud chceš mít vstupy vyřešené opravdu pořádně hned. Pokud ne, jdeme na POC logiku.



# Notes

## Aktuální rozhodnutí
- Projekt vyvíjíme nejdřív nad CSV daty.
- Napojení na Interactive Brokers přijde až později.
- Nejprve potřebujeme ověřit správné načítání dat a logiku výpočtu.

## Logika projektu
- Základní vstup budou historická OHLCV data.
- POC se bude počítat z volume rozloženého přes cenové úrovně.
- Zajímají nás hlavně měsíční a roční POC levely.

## Validita levelu
- Level je platný do chvíle, než ho pozdější cena zasáhne.
- Zásah může být dotyk nebo průraz.
- Přesnou definici zásahu ještě doladíme při implementaci.

## Praktický směr
- Nejprve 1 symbol
- potom více symbolů
- potom export výsledků
- potom IB napojení

## Aktuální stav
- GitHub repo je založené
- repo je naklonované lokálně
- základní kostra projektu funguje
- dokumentace v docs/ už je založená
