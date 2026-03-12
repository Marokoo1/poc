
### 4) docs/plan.md

```bash
cat > docs/plan.md <<'EOF'
# Plan

## Hlavní cíl projektu

Vytvořit Python projekt pro práci s Point of Control (POC) levely nad historickými daty tak, aby šel použít pro:

- vizuální analýzu levelů
- filtrování validních levelů
- historický backtest návratů k POC
- pozdější automatizaci

## Co už projekt umí

- načítat historická data z CSV
- počítat POC levely
- exportovat levely do CSV
- obohacovat levely o další kontext
- zobrazovat levely v dashboardu
- provádět historický backtest
- zobrazovat backtest obchody v dashboardu

## Aktuální vývojová fáze

### Fáze 1 — hotovo
- struktura projektu
- načítání a čištění CSV
- základní výpočet POC
- export levelů

### Fáze 2 — hotovo
- enrichment logika nad levely
- základní validace levelů
- dashboard pro ruční kontrolu

### Fáze 3 — rozpracováno
- realističtější backtest vstupů
- departure threshold
- clean touch
- gap-cross invalidace
- rotation invalidace

### Fáze 4 — další krok
- zpřísnění parametrů validního návratu
- lepší konfigurace parametrů
- další vizuální validace v dashboardu

### Fáze 5 — později
- oddělení parametrů do samostatného configu
- lepší reporting
- případné napojení na další datové zdroje

## Praktický směr

Nejdřív chceme doladit logiku nad lokálními CSV daty.  
Teprve potom dává smysl řešit další automatizaci nebo nové zdroje dat.
EOF
