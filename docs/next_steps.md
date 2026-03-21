
---

# 4) `docs/next_steps.md`

```md
# Next Steps

## Aktuální priorita

Dokončit a ověřit historický backtest pro tyto režimy:

- **POC**
- **IB**
- **POC + IB confluence**

Cílem není jen přidat další levely, ale zjistit:
- jestli jsou IB levely samostatně obchodovatelné
- jestli konfluence POC + IB zlepšuje výsledky oproti samotnému POC
- jestli dává systém smysl pro další paper trading etapu

---

## Bezprostřední úkoly

### 1. Dokončit IB vrstvu
- roční IB
- měsíční IB
- standardní projekce
- volitelný fib režim
- ověřit obousměrné projekce nahoru i dolů

### 2. Zapracovat IB do backtestu
- samostatný režim `POC`
- samostatný režim `IB`
- režim `POC + IB`
- případně i `IB + POC`

### 3. Doladit confluence logiku
- definice vzdálenosti mezi POC a IB
- ATR-based threshold
- ověřit, kdy je confluence skutečně relevantní
- neplést dohromady signal level a confirmační level

### 4. Porovnání výsledků
Porovnat minimálně:
- počet obchodů
- win rate
- průměrný obchod
- drawdown
- profit factor
- expectancy

### 5. Vizualní kontrola
V dashboardu ručně projít:
- POC-only obchody
- IB-only obchody
- POC+IB obchody
- několik problémových případů z grafu

---

## Co nesmí přeskočit pořadí

Nejdřív:
1. správný výpočet levelů
2. správná aktivace a invalidace
3. porovnání režimů
4. ruční kontrola grafů

Až potom:
5. paper trading přes IB Gateway

---

## Další etapa po dokončení tohoto kroku

Jakmile bude historický backtest POC + IB hotový a dává smysl, naváže další pracovní blok:

### IB Gateway / TWS paper trading
- napojení na TWS / IB Gateway
- čtení obchodních signálů ze systému
- tvorba obchodní logiky pro paper účet
- zadávání a správa příkazů
- evidence a kontrola stavu obchodů

---

## Rozhodovací otázka pro další fázi

Do další etapy nepřecházet jen proto, že “už to běží”.
Přejít teprve tehdy, když bude jasné:

- co přesně se bude obchodovat
- jaký level je signal level
- jaký filtr je potvrzující
- jaká je logika vstupu, SL, PT a invalidace
- že výsledky v backtestu dávají smysl i při ruční kontrole grafu
