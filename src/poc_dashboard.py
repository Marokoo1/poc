from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml


PERIOD_STYLES = {
    "weekly": {"label": "Weekly", "color": "#6EC1FF", "width": 1.4, "dash": "dot", "rank": 3},
    "monthly": {"label": "Monthly", "color": "#FFB84D", "width": 2.2, "dash": "dash", "rank": 2},
    "yearly": {"label": "Yearly", "color": "#FF6B6B", "width": 3.2, "dash": "solid", "rank": 1},
}

STANDARD_OHLCV = ["Date", "Open", "High", "Low", "Close", "Volume"]


@dataclass
class DataSource:
    name: str
    raw_data_dir: Path
    processed_data_dir: Path
    raw_file_pattern: str = "{ticker}.csv"
    poc_file_pattern: str = "{ticker}_poc.csv"


@dataclass
class ChartSettings:
    selected_periods: list[str]
    show_only_untouched: bool
    show_labels: bool
    nearest_only: bool
    nearest_count: int
    months_back: int
    extend_from_period_start: bool


@st.cache_data(show_spinner=False, ttl=60)
def load_yaml_settings(settings_path: str) -> dict:
    path = Path(settings_path)
    if not path.exists():
        raise FileNotFoundError(f"Settings file nebyl nalezen: {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def discover_project_root(settings_path: Path) -> Path:
    return settings_path.parent.parent if settings_path.parent.name == "config" else settings_path.parent


def build_sources(settings: dict, project_root: Path) -> tuple[dict[str, DataSource], str]:
    sources: dict[str, DataSource] = {}

    paths_cfg = settings.get("paths", {})
    raw_dir = paths_cfg.get("raw_data_dir", "data/raw")
    processed_dir = paths_cfg.get("processed_data_dir", "data/processed")
    csv_cfg = settings.get("csv", {})

    sources["project_default"] = DataSource(
        name="project_default",
        raw_data_dir=(project_root / raw_dir).resolve(),
        processed_data_dir=(project_root / processed_dir).resolve(),
        raw_file_pattern=csv_cfg.get("file_pattern", "{symbol}.csv").replace("{symbol}", "{ticker}"),
        poc_file_pattern="{ticker}_poc.csv",
    )

    dashboard_cfg = settings.get("dashboard", {})
    sources_cfg = dashboard_cfg.get("data_sources", {})
    default_source = dashboard_cfg.get("default_data_source", "project_default")

    for source_name, source_cfg in sources_cfg.items():
        raw_source_dir = Path(str(source_cfg.get("raw_data_dir", raw_dir)))
        processed_source_dir = Path(str(source_cfg.get("processed_data_dir", processed_dir)))

        if not raw_source_dir.is_absolute():
            raw_source_dir = (project_root / raw_source_dir).resolve()
        if not processed_source_dir.is_absolute():
            processed_source_dir = (project_root / processed_source_dir).resolve()

        sources[source_name] = DataSource(
            name=source_name,
            raw_data_dir=raw_source_dir,
            processed_data_dir=processed_source_dir,
            raw_file_pattern=str(source_cfg.get("raw_file_pattern", csv_cfg.get("file_pattern", "{symbol}.csv"))).replace("{symbol}", "{ticker}"),
            poc_file_pattern=str(source_cfg.get("poc_file_pattern", "{ticker}_poc.csv")),
        )

    if default_source not in sources:
        default_source = "project_default"

    return sources, default_source


@st.cache_data(show_spinner=False, ttl=60)
def list_tickers(processed_dir: str, poc_file_pattern: str) -> list[str]:
    base = Path(processed_dir)
    if not base.exists():
        return []

    suffix = poc_file_pattern.replace("{ticker}", "")
    prefix = poc_file_pattern.split("{ticker}")[0] if "{ticker}" in poc_file_pattern else ""

    tickers: list[str] = []
    for file_path in sorted(base.glob("*.csv")):
        name = file_path.name
        if prefix and not name.startswith(prefix):
            continue
        if suffix and not name.endswith(suffix):
            continue
        ticker = name[len(prefix): len(name) - len(suffix) if suffix else None]
        if ticker:
            tickers.append(ticker.upper())
    return sorted(set(tickers))


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key == "datetime":
            rename_map[col] = "Date"
        elif key == "date":
            rename_map[col] = "Date"
        elif key in {"open", "high", "low", "close", "volume"}:
            rename_map[col] = key.capitalize()
    df = df.rename(columns=rename_map)
    missing = [col for col in STANDARD_OHLCV if col not in df.columns]
    if missing:
        raise ValueError(f"OHLCV soubor nemá očekávané sloupce: {missing}")
    out = df[STANDARD_OHLCV].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=STANDARD_OHLCV).sort_values("Date").reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False, ttl=60)
def load_ohlcv_from_csv(raw_file_path: str) -> pd.DataFrame:
    path = Path(raw_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data soubor nebyl nalezen: {path}")
    df = pd.read_csv(path)
    return normalize_ohlcv(df)


@st.cache_data(show_spinner=False, ttl=60)
def load_poc_data(poc_file_path: str, ticker_hint: str | None = None) -> pd.DataFrame:
    path = Path(poc_file_path)
    if not path.exists():
        raise FileNotFoundError(f"POC soubor nebyl nalezen: {path}")

    df = pd.read_csv(path)

    for col in ["PeriodStart", "PeriodEnd"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "Ticker" not in df.columns:
        inferred_ticker = None
        if ticker_hint:
            inferred_ticker = str(ticker_hint).upper().strip()
        else:
            name = path.stem
            if name.lower().endswith("_poc"):
                inferred_ticker = name[:-4].upper().strip()
            else:
                inferred_ticker = name.upper().strip()
        df["Ticker"] = inferred_ticker

    required = {
        "Ticker", "PeriodType", "Period", "PeriodStart", "PeriodEnd",
        "POC", "POC_Volume", "Period_High", "Period_Low", "Period_Close"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"POC CSV nemá očekávané sloupce: {sorted(missing)}")

    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["PeriodType"] = df["PeriodType"].astype(str).str.lower().str.strip()
    df["POC"] = pd.to_numeric(df["POC"], errors="coerce")
    df["POC_Volume"] = pd.to_numeric(df["POC_Volume"], errors="coerce")
    df["Period_High"] = pd.to_numeric(df["Period_High"], errors="coerce")
    df["Period_Low"] = pd.to_numeric(df["Period_Low"], errors="coerce")
    df["Period_Close"] = pd.to_numeric(df["Period_Close"], errors="coerce")

    return df.dropna(subset=["Ticker", "PeriodType", "POC", "PeriodStart", "PeriodEnd"]).copy()


@st.cache_data(show_spinner=False, ttl=60)
def enrich_levels(levels: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    if levels.empty:
        return levels.copy()

    out = levels.copy()
    ohlcv_idx = ohlcv.copy().set_index("Date")
    last_close = float(ohlcv_idx["Close"].iloc[-1])
    out["LastPrice"] = last_close
    out["SignedDist"] = (last_close - out["POC"]).round(4)
    out["AbsDist"] = out["SignedDist"].abs().round(4)

    touched_values: list[bool] = []
    touch_dates: list[pd.Timestamp | pd.NaT] = []

    for _, row in out.iterrows():
        after_end = ohlcv_idx[ohlcv_idx.index > pd.Timestamp(row["PeriodEnd"])]
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
    return out.sort_values(["AbsDist", "PeriodRank", "PeriodStart"], ascending=[True, True, False]).reset_index(drop=True)


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
    ohlcv_idx = ohlcv.copy().set_index("Date")
    fig = go.Figure()

    history_start = ohlcv_idx.index.max() - pd.DateOffset(months=settings.months_back)
    chart_df = ohlcv_idx[ohlcv_idx.index >= history_start].copy()
    if chart_df.empty:
        chart_df = ohlcv_idx.copy()

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
        yaxis=dict(title="Cena", side="right", showgrid=True, gridcolor="rgba(148,163,184,0.10)", zeroline=False),
    )
    return fig


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    table = df[[
        "PeriodType", "Period", "POC", "LastPrice", "SignedDist", "AbsDist",
        "Valid", "Touched", "TouchDate", "PeriodStart", "PeriodEnd", "POC_Volume"
    ]].copy()
    return table.rename(columns={
        "PeriodType": "Type",
        "LastPrice": "Last",
        "SignedDist": "Dist",
        "POC_Volume": "POC_Vol",
        "PeriodStart": "Start",
        "PeriodEnd": "End",
    })


def metric_block(levels: pd.DataFrame) -> tuple[int, int, float | None]:
    total = len(levels)
    valid = int(levels["Valid"].sum()) if not levels.empty else 0
    nearest = float(levels["SignedDist"].iloc[0]) if not levels.empty else None
    return total, valid, nearest


def resolve_source_files(source: DataSource, ticker: str) -> tuple[Path, Path]:
    raw_path = source.raw_data_dir / source.raw_file_pattern.format(ticker=ticker)
    poc_path = source.processed_data_dir / source.poc_file_pattern.format(ticker=ticker)
    return raw_path, poc_path


def sidebar_controls(sources: dict[str, DataSource], default_source: str, tickers: Iterable[str]) -> tuple[str, str, ChartSettings]:
    st.sidebar.header("Nastavení")
    source_name = st.sidebar.selectbox("Zdroj dat", options=list(sources.keys()), index=list(sources.keys()).index(default_source))
    ticker = st.sidebar.selectbox("Ticker", options=list(tickers))

    selected_periods = st.sidebar.multiselect("Zobrazit periody", ["weekly", "monthly", "yearly"], default=["monthly", "yearly"])
    show_only_untouched = st.sidebar.checkbox("Pouze validní / dosud netestované levely", value=False)
    nearest_only = st.sidebar.checkbox("Zobrazit jen nejbližší levely", value=True)
    nearest_count = st.sidebar.slider("Počet nejbližších levelů", 1, 12, 6)
    months_back = st.sidebar.slider("Kolik měsíců historie v grafu", 3, 36, 12)
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
    return source_name, ticker, settings


def main() -> None:
    st.set_page_config(page_title="POC Dashboard", layout="wide")
    st.title("POC Dashboard")
    st.caption("Čte lokální raw a processed data projektu. Žádné dvojí stahování cen.")

    default_settings_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    settings_path = Path(st.sidebar.text_input("Settings YAML", value=str(default_settings_path)))

    try:
        settings = load_yaml_settings(str(settings_path))
        project_root = discover_project_root(settings_path)
        sources, default_source = build_sources(settings, project_root)
    except Exception as exc:
        st.error(f"Nepodařilo se načíst YAML konfiguraci: {exc}")
        st.stop()

    tickers = list_tickers(str(sources[default_source].processed_data_dir), sources[default_source].poc_file_pattern)
    if not tickers:
        st.error(f"V processed složce nejsou nalezené žádné *_poc.csv soubory: {sources[default_source].processed_data_dir}")
        st.stop()

    source_name, ticker, chart_settings = sidebar_controls(sources, default_source, tickers)
    source = sources[source_name]
    raw_path, poc_path = resolve_source_files(source, ticker)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Raw: `{raw_path}`")
    st.sidebar.caption(f"POC: `{poc_path}`")

    ohlcv = None
    poc_levels = None

    try:
        ohlcv = load_ohlcv_from_csv(str(raw_path))
    except Exception as exc:
        st.error(f"Nepodařilo se načíst raw data pro {ticker}: {exc}")
        st.stop()

    try:
        poc_levels = load_poc_data(str(poc_path), ticker_hint=ticker)
    except Exception as exc:
        st.error(f"Nepodařilo se načíst POC data pro {ticker}: {exc}")
        st.stop()

    if ohlcv is None:
        st.error("OHLCV data nebyla načtena.")
        st.stop()

    if poc_levels is None:
        st.error("POC data nebyla načtena.")
        st.stop()

    ticker_levels = poc_levels[poc_levels["Ticker"] == ticker].copy()
    if ticker_levels.empty:
        st.warning(f"Pro ticker {ticker} nebyly v POC souboru nalezeny žádné levely.")
        st.stop()

    enriched = enrich_levels(ticker_levels, ohlcv)
    filtered = filter_levels(enriched, chart_settings)

    total, valid_count, nearest_dist = metric_block(filtered)
    c1, c2, c3 = st.columns(3)
    c1.metric("Zobrazené levely", total)
    c2.metric("Validní levely", valid_count)
    c3.metric("Nejbližší distance", "—" if nearest_dist is None else f"{nearest_dist:+.2f}")

    if not chart_settings.selected_periods:
        st.info("Vyber alespoň jednu periodu v levém panelu.")
        st.stop()
    if filtered.empty:
        st.warning("Po zvolených filtrech nezůstal žádný level.")
        st.stop()

    st.plotly_chart(build_chart(ohlcv, filtered, ticker, chart_settings), use_container_width=True)

    st.subheader("Tabulka levelů")
    table = format_table(filtered)
    st.dataframe(
        table,
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

    st.download_button(
        "Stáhnout aktuálně filtrovanou tabulku",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_poc_levels_filtered.csv",
        mime="text/csv",
    )

    with st.expander("Jak číst distance"):
        st.write(
            "Distance je počítaná jako poslední cena mínus POC. Kladná hodnota znamená, že aktuální cena je nad levelem. "
            "Záporná hodnota znamená, že cena je pod levelem."
        )


if __name__ == "__main__":
    main()
