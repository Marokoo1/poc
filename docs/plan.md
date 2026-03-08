poc/
├── README.md
├── .gitignore
├── requirements.txt
├── config/
│   └── settings.yaml
├── data/
│   ├── raw/
│   └── processed/
└── src/
    ├── main.py
    ├── config.py
    ├── data_fetcher.py
    ├── poc_calculator.py
    └── utils.py




# Plan

## Hlavní cíl
Vytvořit Python projekt pro práci s Point of Control (POC) levely nad historickými daty.

## Co má projekt postupně umět
- načíst historická data z CSV
- později načítat historická data z Interactive Brokers
- počítat měsíční POC
- počítat roční POC
- určovat, zda je level stále validní
- exportovat výsledky do CSV
- později případně přidat automatické spuštění

## Doporučené pořadí vývoje

### Fáze 1 – základ projektu
- vytvořit strukturu projektu
- připravit config
- ověřit spuštění programu

### Fáze 2 – načítání a čištění CSV
- načítat data ze souborů CSV
- kontrolovat povinné sloupce
- převést datum na správný formát
- seřadit data podle data

### Fáze 3 – výpočet POC
- spočítat první měsíční POC
- spočítat první roční POC
- umět vrátit poslední 3 měsíční POC
- umět vrátit poslední 3 roční POC

### Fáze 4 – validita levelů
- level platí do prvního dotyku nebo průrazu cenou
- po zásahu se level označí jako neplatný

### Fáze 5 – export výsledků
- uložit výsledné levely do CSV
- připravit výstup pro další použití

### Fáze 6 – napojení na IB
- připojení k TWS / IB Gateway
- stažení historických dat
- uložení do lokálních souborů
- použití stejné POC logiky jako pro CSV

## Poznámka k vývoji
Nejdřív budeme ladit logiku na CSV datech.
Napojení na IB přijde až ve chvíli, kdy bude výpočet POC fungovat správně.
