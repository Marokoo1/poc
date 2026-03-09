"""
POC Dashboard - přehledný interaktivní dashboard pro Point of Control levely.

Spuštění:
    pip install streamlit plotly yfinance pandas
    streamlit run poc_dashboard.py

Co umí:
- vybrat ticker
- zapnout/vypnout weekly / monthly / yearly levely
- filtrovat pouze validní (dosud netestované) levely
- omezit počet zobrazených nejbližších levelů k aktuální ceně
- zapnout/vypnout popisky levelů
- měnit délku historie grafu
- zobrazit tabulku levelů se signed distance (kladné = cena nad levelem, záporné = cena pod levelem)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ============================================================
# Nastavení
# ============================================================

DEFAULT_POC_FILE = "poc_output.csv"
DEFAULT_HISTORY_START = "2024-01-01"

PERIOD_STYLES = {
    "weekly": {
        "label": "Weekly",
        "color": "#6EC1FF",
        "width": 1.4,
        "dash": "dot",
        "rank": 3,
    },
    "monthly": {
        "label": "Monthly",
        "color": "#FFB84D",
        "width": 2.2,
        "dash": "dash",
        "rank": 2,
    },
    "yearly": {
        "label": "Yearly",
        "color": "#FF6B6B",
        "width": 3.2,
        "dash": "solid",
        "rank": 1,
    },
}


@dataclass
class ChartSettings:
    selected_periods: list[str]
    show_only_untouched: bool
    show_labels: bool
    nearest_only: bool
    nearest_count: int
    months_back: int
    extend_from_period_start: bool


@st.cache_data(show_spinner=False, ttl=3600)
def load_poc_data(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Soubor nebyl nalezen: {path.resolve()}")

    df = pd.read_csv(path, parse_dates=["PeriodStart", "PeriodEnd"])
    required = {
        "Ticker", "PeriodType", "Period", "PeriodStart", "PeriodEnd",
        "POC", "POC_Volume", "Period_High", "Period_Low", "Period_Close"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV nemá očekávané sloupce: {sorted(missing)}")

    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["PeriodType"] = df["PeriodType"].astype(str).str.lower().str.strip()
    df["POC"] = pd.to_numeric(df["POC"], errors="coerce")
    df = df.dropna(subset=["Ticker", "PeriodType", "POC", "PeriodStart", "PeriodEnd"]).copy()
    return df.sort_values(["Ticker", "PeriodStart", "PeriodType"]).reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=3600)
def load_ohlcv(ticker: str, start: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if raw.empty:
        return raw
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index)
    return raw[["Open", "High", "Low", "Close", "Volume"]].dropna()


@st.cache_data(show_spinner=False, ttl=3600)
def enrich_levels(levels: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    if levels.empty:
        return levels.copy()

    out = levels.copy()
    last_close = float(ohlcv["Close"].iloc[-1])
    out["LastPrice"] = last_close
    out["SignedDist"] = (last_close - out["POC"]).round(4)  # + = cena nad levelem
    out["AbsDist"] = out["SignedDist"].abs().round(4)

    touched_values: list[bool] = []
    touch_dates: list[pd.Timestamp | pd.NaT] = []

    for _, row in out.iterrows():
        after_end = ohlcv[ohlcv.index > pd.Timestamp(row["PeriodEnd"])]
        touched_mask = (after_end["Low"] <= row["POC"]) & (after_end["High"] >= row["POC"])
        if touched_mask.any():
            first_touch = after_end.index[touched_mask][0]
            touched_values.append(True)
            touch_dates.append(first_touch)
        else:
            touched_values.append(False)
            touch_dates.append(pd.NaT)

    out["Touched"] = touched_values
    out["Valid"] = ~out["Touched"]
    out["TouchDate"] = touch_dates
    out["AgeDays"] = (pd.Timestamp.today().normalize() - out["PeriodEnd"]).dt.days
    out["PeriodRank"] = out["PeriodType"].map(lambda x: PERIOD_STYLES.get(x, {}).get("rank", 99))
    out = out.sort_values(["AbsDist", "PeriodRank", "PeriodStart"], ascending=[True, True, False]).reset_index(drop=True)
    return out


def filter_levels(levels: pd.DataFrame, settings: ChartSettings) -> pd.DataFrame:
    if levels.empty:
        return levels.copy()

    data = levels[levels["PeriodType"].isin(settings.selected_periods)].copy()
    if settings.show_only_untouched:
        data = data[data["Valid"]]
    if settings.nearest_only:
        data = data.nsmallest(settings.nearest_count, ["AbsDist", "PeriodRank"])
    return data.sort_values(["PeriodRank", "AbsDist", "PeriodStart"], ascending=[True, True, False]).reset_index(drop=True)



def build_chart(ohlcv: pd.DataFrame, levels: pd.DataFrame, ticker: str, settings: ChartSettings) -> go.Figure:
    fig = go.Figure()

    history_start = ohlcv.index.max() - pd.DateOffset(months=settings.months_back)
    chart_df = ohlcv[ohlcv.index >= history_start].copy()
    if chart_df.empty:
        chart_df = ohlcv.copy()

    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df["Open"],
        high=chart_df["High"],
        low=chart_df["Low"],
        close=chart_df["Close"],
        name=ticker,
        increasing_line_color="#22C55E",
        decreasing_line_color="#EF4444",
        whiskerwidth=0.35,
        showlegend=False,
    ))

    legend_done: set[str] = set()

    for _, row in levels.iterrows():
        style = PERIOD_STYLES[row["PeriodType"]]
        x0 = pd.Timestamp(row["PeriodStart"]) if settings.extend_from_period_start else pd.Timestamp(row["PeriodEnd"])
        if x0 < chart_df.index.min():
            x0 = chart_df.index.min()

        opacity = 0.95 if bool(row["Valid"]) else 0.35
        label_suffix = "valid" if bool(row["Valid"]) else "tested"

        fig.add_trace(go.Scatter(
            x=[x0, chart_df.index.max()],
            y=[row["POC"], row["POC"]],
            mode="lines",
            line=dict(color=style["color"], width=style["width"], dash=style["dash"]),
            opacity=opacity,
            name=style["label"],
            legendgroup=row["PeriodType"],
            showlegend=row["PeriodType"] not in legend_done,
            hovertemplate=(
                f"<b>{style['label']} POC</b><br>"
                f"Period: {row['Period']}<br>"
                f"POC: {row['POC']:.2f}<br>"
                f"Distance: {row['SignedDist']:+.2f}<br>"
                f"Status: {label_suffix}<br>"
                "<extra></extra>"
            ),
        ))
        legend_done.add(row["PeriodType"])

        if settings.show_labels:
            fig.add_annotation(
                x=chart_df.index.max(),
                y=row["POC"],
                text=f"{style['label']} {row['POC']:.2f}",
                showarrow=False,
                xanchor="left",
                xshift=8,
                font=dict(size=11, color=style["color"]),
                bgcolor="rgba(15,23,42,0.72)",
                bordercolor=style["color"],
                borderwidth=1,
            )

    last_close = float(chart_df["Close"].iloc[-1])
    fig.add_hline(
        y=last_close,
        line_width=1,
        line_dash="solid",
        line_color="#E5E7EB",
        opacity=0.45,
        annotation_text=f"Last {last_close:.2f}",
        annotation_position="top left",
    )

    fig.update_layout(
        title=dict(text=f"<b>{ticker}</b> – POC dashboard", x=0.02),
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220",
        height=760,
        margin=dict(l=20, r=140 if settings.show_labels else 30, t=70, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor="rgba(148,163,184,0.10)",
            rangebreaks=[dict(bounds=["sat", "mon"])],
        ),
        yaxis=dict(
            title="Cena",
            side="right",
            showgrid=True,
            gridcolor="rgba(148,163,184,0.10)",
            zeroline=False,
        ),
    )
    return fig



def format_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    table = df[[
        "PeriodType", "Period", "POC", "LastPrice", "SignedDist", "AbsDist",
        "Valid", "Touched", "TouchDate", "PeriodStart", "PeriodEnd", "POC_Volume"
    ]].copy()
    table = table.rename(columns={
        "PeriodType": "Type",
        "Period": "Period",
        "POC": "POC",
        "LastPrice": "Last",
        "SignedDist": "Dist",
        "AbsDist": "AbsDist",
        "Valid": "Valid",
        "Touched": "Touched",
        "TouchDate": "TouchDate",
        "PeriodStart": "Start",
        "PeriodEnd": "End",
        "POC_Volume": "POC_Vol",
    })
    return table



def metric_block(levels: pd.DataFrame) -> tuple[int, int, float | None]:
    total = len(levels)
    valid = int(levels["Valid"].sum()) if not levels.empty else 0
    nearest = float(levels["SignedDist"].iloc[0]) if not levels.empty else None
    return total, valid, nearest



def sidebar_controls(all_tickers: Iterable[str]) -> tuple[str, str, ChartSettings]:
    st.sidebar.header("Nastavení")
    csv_path = st.sidebar.text_input("CSV soubor", value=DEFAULT_POC_FILE)
    ticker = st.sidebar.selectbox("Ticker", options=list(all_tickers))

    selected_periods = st.sidebar.multiselect(
        "Zobrazit periody",
        options=["weekly", "monthly", "yearly"],
        default=["monthly", "yearly"],
    )

    show_only_untouched = st.sidebar.checkbox("Pouze validní / dosud netestované levely", value=False)
    nearest_only = st.sidebar.checkbox("Zobrazit jen nejbližší levely", value=True)
    nearest_count = st.sidebar.slider("Počet nejbližších levelů", min_value=1, max_value=12, value=6)
    months_back = st.sidebar.slider("Kolik měsíců historie v grafu", min_value=3, max_value=36, value=12)
    show_labels = st.sidebar.checkbox("Popisky levelů vpravo", value=True)
    extend_from_period_start = st.sidebar.checkbox("Vést čáru od začátku periody", value=False)

    settings = ChartSettings(
        selected_periods=selected_periods,
        show_only_untouched=show_only_untouched,
        show_labels=show_labels,
        nearest_only=nearest_only,
        nearest_count=nearest_count,
        months_back=months_back,
        extend_from_period_start=extend_from_period_start,
    )
    return csv_path, ticker, settings



def main() -> None:
    st.set_page_config(page_title="POC Dashboard", layout="wide")
    st.title("POC Dashboard")
    st.caption("Přehledné zobrazení Point of Control levelů s filtry a rychlým přepínáním.")

    default_path = Path(DEFAULT_POC_FILE)
    try:
        poc_data = load_poc_data(str(default_path))
    except Exception as exc:
        st.error(f"Nepodařilo se načíst výchozí CSV: {exc}")
        st.stop()

    csv_path, ticker, settings = sidebar_controls(sorted(poc_data["Ticker"].unique().tolist()))

    try:
        poc_data = load_poc_data(csv_path)
    except Exception as exc:
        st.error(f"Nepodařilo se načíst CSV: {exc}")
        st.stop()

    ticker_levels = poc_data[poc_data["Ticker"] == ticker].copy()
    if ticker_levels.empty:
        st.warning(f"Pro ticker {ticker} nejsou v CSV žádné levely.")
        st.stop()

    ohlcv = load_ohlcv(ticker, DEFAULT_HISTORY_START)
    if ohlcv.empty:
        st.error(f"Nepodařilo se stáhnout cenová data pro {ticker}.")
        st.stop()

    enriched = enrich_levels(ticker_levels, ohlcv)
    filtered = filter_levels(enriched, settings)

    total, valid_count, nearest_dist = metric_block(filtered)
    c1, c2, c3 = st.columns(3)
    c1.metric("Zobrazené levely", total)
    c2.metric("Validní levely", valid_count)
    c3.metric("Nejbližší distance", "—" if nearest_dist is None else f"{nearest_dist:+.2f}")

    if not settings.selected_periods:
        st.info("Vyber alespoň jednu periodu v levém panelu.")
        st.stop()

    if filtered.empty:
        st.warning("Po zvolených filtrech nezůstal žádný level.")
        st.stop()

    chart = build_chart(ohlcv, filtered, ticker, settings)
    st.plotly_chart(chart, use_container_width=True)

    st.subheader("Tabulka levelů")
    st.dataframe(
        format_table(filtered),
        use_container_width=True,
        hide_index=True,
        column_config={
            "POC": st.column_config.NumberColumn(format="%.2f"),
            "Last": st.column_config.NumberColumn(format="%.2f"),
            "Dist": st.column_config.NumberColumn(format="%+.2f"),
            "AbsDist": st.column_config.NumberColumn(format="%.2f"),
            "POC_Vol": st.column_config.NumberColumn(format="%.0f"),
            "TouchDate": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Start": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "End": st.column_config.DateColumn(format="YYYY-MM-DD"),
        },
    )

    csv_export = format_table(filtered).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Stáhnout aktuálně filtrovanou tabulku",
        data=csv_export,
        file_name=f"{ticker}_poc_levels_filtered.csv",
        mime="text/csv",
    )

    with st.expander("Jak číst distance"):
        st.write(
            "Distance je počítaná jako poslední cena mínus POC. "
            "Kladná hodnota znamená, že aktuální cena je nad levelem. "
            "Záporná hodnota znamená, že cena je pod levelem."
        )


if __name__ == "__main__":
    main()
