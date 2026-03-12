
### 3) docs/next_steps.md

```bash
cat > docs/next_steps.md <<'EOF'
# Plan na příště

## Hlavní cíl

Doladit parametry a prezentaci backtestu tak, aby obchody lépe odpovídaly tomu, co je vidět v grafu.

## Co už je hotové

- backtest běží
- dashboard pro backtest běží
- přidán departure threshold
- přidán gap-cross filtr
- přidán rotation filtr
- výrazně ubyly slepé vstupy přímo na POC

## Priorita pro další session

### 1. Zpřísnit validitu levelu
- zvýšit minimální odchod ceny od POC
- otestovat výrazně vyšší `activation_threshold_value`
- porovnat weekly / monthly / yearly zvlášť

### 2. Upravovat parametry na jednom místě
- sjednotit nastavení:
  - departure threshold
  - SL
  - TP
  - max hold bars
- držet vše přehledně v `PERIOD_PARAMS`
- případně později přesunout do samostatného configu

### 3. Projít problematické obchody v dashboardu
- najít další případy, které pořád nevypadají realisticky
- ověřit, zda nejde o:
  - příliš malý departure threshold
  - stále moc volný clean touch
  - příliš krátký nebo příliš těsný trade management

### 4. Zlepšit dashboard
- zobrazit jasněji důvod neotevření levelu:
  - `no_departure`
  - `gap_cross`
  - `rotation`
  - `no_touch`
- přidat lepší filtrování podle `exit_reason`

## Konkrétní první krok příště

1. otevřít `src/poc_backtest.py`
2. zvýšit parametry v `PERIOD_PARAMS`
3. znovu spustit:
   ```bash
   python3 src/poc_backtest.py
