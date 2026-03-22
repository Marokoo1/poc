from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from ib_calculator import calculate_all_ib_levels


STANDARD_OHLCV = ["Date", "Open", "High", "Low", "Close", "Volume"]

DISPLAY_MODE_OPTIONS = ["POC", "IB", "POC + IB"]
POC_PERIODS = ["weekly", "monthly", "yearly"]
IB_PERIODS = ["monthly_ib", "yearly_ib", "monthly_ib_std", "yearly_ib_std", "monthly_ib_fib", "yearly_ib_fib"]
IB_FAMILIES = ["ib_core", "ib_standard", "ib_fib"]

STYLE_MAP = {
    "weekly": {"label": "Weekly POC", "color": "#6EC1FF", "width": 1.4, "dash": "dot", "rank": 6},
    "monthly": {"label": "Monthly POC", "color": "#FFB84D", "width": 2.2, "dash": "dash", "rank": 5},
    "yearly": {"label": "Yearly POC", "color": "#FF6B6B", "width": 3.0, "dash": "solid", "rank": 4},
    "monthly_ib": {"label": "Monthly IB Core", "color": "#34D399", "width": 2.4, "dash": "solid", "rank": 3},
    "yearly_ib": {"label": "Yearly IB Core", "color": "#10B981", "width": 3.2, "dash": "solid", "rank": 2},
    "monthly_ib_std": {"label": "Monthly IB Std", "color": "#A78BFA", "width": 1.8, "dash": "dash", "rank": 7},
    "yearly_ib_std": {"label": "Yearly IB Std", "color": "#8B5CF6", "width": 2.2, "dash": "dash", "rank": 1},
    "monthly_ib_fib": {"label": "Monthly IB Fib", "color": "#F472B6", "width": 1.6, "dash": "dot", "rank": 9},
    "yearly_ib_fib": {"label": "Yearly IB Fib", "color": "#EC4899", "width": 2.0, "dash": "dot", "rank": 8},
}


@dataclass
class DataSource:
    name: str
    raw_data_dir: Path
    processed_data_dir: Path
    raw_file_pattern: str = "{ticker}.csv"
    poc_file_pattern: str = "{ticker}_poc.csv"


@dataclass
class IBSettings:
    enabled: bool
    yearly_enabled: bool
    monthly_enabled: bool
    monthly_mode: str
    standard_projection_enabled: bool
    standard_multipliers: list[float]
    fibonacci_projection_enabled: bool
    fibonacci_multipliers: list[float]


@dataclass
class ChartSettings:
    display_mode: str
    selected_poc_periods: list[str]
    selected_ib_periods: list[str]
    selected_years: list[int]
    selected_months: list[int]
    level_status: str
    show_labels: bool
    nearest_only: bool
    nearest_count: int
    months_back: int
    extend_from_activation: bool
    show_confluence_only: bool
    confluence_max_atr: float




def enrich_set_time_columns(levels: pd.DataFrame) -> pd.DataFrame:
    if levels.empty:
        out = levels.copy()
        out["SetYear"] = pd.Series(dtype="Int64")
        out["SetMonth"] = pd.Series(dtype="Int64")
        out["SetLabel"] = pd.Series(dtype="string")
        out["IBGroup"] = pd.Series(dtype="string")
        out["LevelFamily"] = out.get("LevelFamily", pd.Series(dtype="string"))
        return out

    out = levels.copy()
    base_time = pd.to_datetime(out.get("PeriodStart"), errors="coerce")
    if "AnchorStart" in out.columns:
        anchor_time = pd.to_datetime(out["AnchorStart"], errors="coerce")
        base_time = base_time.fillna(anchor_time)
    if "ActiveFrom" in out.columns:
        active_time = pd.to_datetime(out["ActiveFrom"], errors="coerce")
        base_time = base_time.fillna(active_time)

    out["SetYear"] = base_time.dt.year.astype("Int64")
    out["SetMonth"] = base_time.dt.month.astype("Int64")

    def make_set_label(row: pd.Series) -> str:
        period_type = str(row.get("PeriodType", "")).lower()
        year = row.get("SetYear")
        month = row.get("SetMonth")
        period = str(row.get("Period", "")).strip()
        if pd.notna(year) and "monthly" in period_type and pd.notna(month):
            return f"{int(year):04d}-{int(month):02d}"
        if pd.notna(year) and "yearly" in period_type:
            return f"{int(year):04d}"
        return period

    out["SetLabel"] = out.apply(make_set_label, axis=1)

    def make_ib_group(row: pd.Series) -> str:
        if str(row.get("Source", "")).upper() != "IB":
            return ""
        period_type = str(row.get("PeriodType", "")).lower()
        prefix = "Y" if "yearly" in period_type else "M"
        return f"{prefix} {row.get('SetLabel', '')}".strip()

    out["IBGroup"] = out.apply(make_ib_group, axis=1)
    out["LevelFamily"] = np.where(out["Source"].eq("IB"), "ib", out.get("LevelFamily", "poc"))
    return out

@st.cache_data(show_spinner=False, ttl=60)
def load_yaml_settings(settings_path: str) -> dict:
    path = Path(settings_path)
    if not path.exists():
        raise FileNotFoundError(f"Settings file nebyl nalezen: {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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
        inferred_ticker = str(ticker_hint or path.stem.replace("_poc", "")).upper().strip()
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
    numeric_cols = ["POC", "POC_Volume", "Period_High", "Period_Low", "Period_Close"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["Ticker", "PeriodType", "POC", "PeriodStart", "PeriodEnd"]).copy()


@st.cache_data(show_spinner=False, ttl=60)
def compute_ib_levels_from_ohlcv(raw_file_path: str, ticker: str, ib_settings_dict: dict) -> pd.DataFrame:
    ohlcv = load_ohlcv_from_csv(raw_file_path)
    ib_df = calculate_all_ib_levels(ohlcv.copy(), ticker=ticker, settings=ib_settings_dict)
    if ib_df.empty:
        return ib_df
    return ib_df


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
        if key in {"datetime", "date"}:
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


def infer_ib_side(level_name: str) -> str:
    name = str(level_name).upper()
    if name.endswith("_DN") or name.endswith("IBL"):
        return "support"
    return "resistance"


def prepare_poc_levels(levels: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    if levels.empty:
        return pd.DataFrame()
    out = levels.copy()
    ohlcv_idx = ohlcv.copy().set_index("Date")
    last_close = float(ohlcv_idx["Close"].iloc[-1])
    out["Source"] = "POC"
    out["LevelFamily"] = "poc"
    out["LevelName"] = out["PeriodType"].astype(str).str.upper() + "_POC"
    out["LevelPrice"] = out["POC"]
    out["AnchorStart"] = out["PeriodEnd"]
    out["AnchorEnd"] = pd.NaT
    out["TouchDate"] = pd.NaT
    touched_values: list[bool] = []
    touch_dates: list[pd.Timestamp | pd.NaT] = []
    for _, row in out.iterrows():
        after_end = ohlcv_idx[ohlcv_idx.index > pd.Timestamp(row["PeriodEnd"])]
        touched_mask = (after_end["Low"] <= row["POC"]) & (after_end["High"] >= row["POC"])
        if touched_mask.any():
            first_touch = pd.Timestamp(after_end.index[touched_mask][0])
            touched_values.append(True)
            touch_dates.append(first_touch)
        else:
            touched_values.append(False)
            touch_dates.append(pd.NaT)
    out["Touched"] = touched_values
    out["Valid"] = ~out["Touched"]
    out["TouchDate"] = touch_dates
    out["Status"] = out["Valid"].map(lambda x: "active" if bool(x) else "tested")
    out["DisplayUntil"] = out["TouchDate"].where(out["TouchDate"].notna(), pd.NaT)
    out["LastPrice"] = last_close
    out["SignedDist"] = (last_close - out["LevelPrice"]).round(4)
    out["AbsDist"] = out["SignedDist"].abs().round(4)
    out["PeriodRank"] = out["PeriodType"].map(lambda x: STYLE_MAP.get(x, {}).get("rank", 99))
    return out


def prepare_ib_levels(levels: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    if levels.empty:
        return pd.DataFrame()
    prices = ohlcv.copy().sort_values("Date").reset_index(drop=True)
    last_close = float(prices["Close"].iloc[-1])
    out = levels.copy()
    out["Source"] = "IB"
    out["AnchorStart"] = pd.to_datetime(out.get("ActiveFrom"), errors="coerce")
    out["LevelPrice"] = pd.to_numeric(out["LevelPrice"], errors="coerce")
    out["LevelSide"] = out["LevelName"].apply(infer_ib_side)

    tested_at: list[pd.Timestamp | pd.NaT] = []
    touches: list[int] = []
    for _, row in out.iterrows():
        active_from = row.get("AnchorStart")
        level_price = row.get("LevelPrice")
        side = row.get("LevelSide")
        if pd.isna(active_from) or pd.isna(level_price):
            tested_at.append(pd.NaT)
            touches.append(0)
            continue
        future = prices[prices["Date"] >= active_from]
        if future.empty:
            tested_at.append(pd.NaT)
            touches.append(0)
            continue
        if side == "resistance":
            hits = future[future["High"] >= level_price]
        else:
            hits = future[future["Low"] <= level_price]
        if hits.empty:
            tested_at.append(pd.NaT)
            touches.append(0)
        else:
            tested_at.append(pd.Timestamp(hits.iloc[0]["Date"]))
            touches.append(int(len(hits)))

    out["TouchDate"] = tested_at
    out["Touched"] = out["TouchDate"].notna()
    out["Valid"] = ~out["Touched"]
    out["Status"] = out["Valid"].map(lambda x: "active" if bool(x) else "tested")
    out["TouchesCount"] = touches
    out["DisplayUntil"] = out["TouchDate"].where(out["TouchDate"].notna(), pd.NaT)
    out["LastPrice"] = last_close
    out["SignedDist"] = (last_close - out["LevelPrice"]).round(4)
    out["AbsDist"] = out["SignedDist"].abs().round(4)
    out["PeriodRank"] = out["PeriodType"].map(lambda x: STYLE_MAP.get(x, {}).get("rank", 99))
    return out


def filter_unified_levels(levels: pd.DataFrame, settings: ChartSettings) -> pd.DataFrame:
    if levels.empty:
        return levels.copy()
    data = levels.copy()

    allowed_sources: list[str]
    if settings.display_mode == "POC":
        allowed_sources = ["POC"]
    elif settings.display_mode == "IB":
        allowed_sources = ["IB"]
    else:
        allowed_sources = ["POC", "IB"]
    data = data[data["Source"].isin(allowed_sources)]

    if "POC" in allowed_sources:
        poc_mask = data["Source"].eq("POC")
        data = pd.concat([
            data[poc_mask & data["PeriodType"].isin(settings.selected_poc_periods)],
            data[~poc_mask],
        ], ignore_index=True)

    if "IB" in allowed_sources:
        ib_mask = data["Source"].eq("IB")
        ib_data = data[ib_mask].copy()
        if settings.selected_ib_periods:
            ib_data = ib_data[ib_data["PeriodType"].isin(settings.selected_ib_periods)]
        data = pd.concat([data[~ib_mask], ib_data], ignore_index=True)

    if settings.selected_years and "SetYear" in data.columns:
        data = data[data["SetYear"].isin(settings.selected_years)]

    if settings.selected_months and "SetMonth" in data.columns:
        data = data[data["SetMonth"].isin(settings.selected_months)]

    if settings.level_status == "Only active":
        data = data[data["Valid"]]
    elif settings.level_status == "Only tested":
        data = data[~data["Valid"]]

    if settings.display_mode == "POC + IB" and settings.show_confluence_only:
        data = apply_confluence_filter(data, settings.confluence_max_atr)

    if settings.nearest_only and not data.empty:
        data = data.nsmallest(settings.nearest_count, ["AbsDist", "PeriodRank"])

    return data.sort_values(["Source", "PeriodRank", "AbsDist", "AnchorStart"], ascending=[True, True, True, False]).reset_index(drop=True)


def apply_confluence_filter(levels: pd.DataFrame, max_atr: float) -> pd.DataFrame:
    if levels.empty:
        return levels.copy()
    poc = levels[levels["Source"] == "POC"].copy()
    ib = levels[levels["Source"] == "IB"].copy()
    if poc.empty or ib.empty:
        return pd.DataFrame(columns=levels.columns)

    all_prices = pd.concat([poc["LevelPrice"], ib["LevelPrice"]], ignore_index=True)
    price_scale = float(all_prices.std()) if len(all_prices) > 1 else 0.0
    if not pd.notna(price_scale) or price_scale <= 0:
        price_scale = max(float(all_prices.abs().median()) * 0.01, 0.01)
    max_abs_distance = max_atr * price_scale

    keep_poc = set()
    keep_ib = set()
    for poc_idx, poc_row in poc.iterrows():
        distances = (ib["LevelPrice"] - float(poc_row["LevelPrice"])).abs()
        matched = ib[distances <= max_abs_distance]
        if not matched.empty:
            keep_poc.add(poc_idx)
            keep_ib.update(matched.index.tolist())

    return pd.concat([poc.loc[sorted(keep_poc)], ib.loc[sorted(keep_ib)]], ignore_index=True)


def format_level_badge(row: pd.Series) -> str:
    source = str(row.get("Source", "")).upper()
    period_type = str(row.get("PeriodType", "")).lower()
    period = str(row.get("Period", "")).strip()
    level_name = str(row.get("LevelName", "")).upper()

    if source == "IB":
        if "yearly" in period_type:
            bucket = "Y"
        else:
            bucket = "M"

        if level_name.endswith("IBH") or level_name.endswith("IBL"):
            proj = "0"
        elif "_100_UP" in level_name:
            proj = "100"
        elif "_150_UP" in level_name:
            proj = "150"
        elif "_200_UP" in level_name:
            proj = "200"
        elif "_300_UP" in level_name:
            proj = "300"
        elif "_100_DN" in level_name:
            proj = "-100"
        elif "_150_DN" in level_name:
            proj = "-150"
        elif "_200_DN" in level_name:
            proj = "-200"
        elif "_300_DN" in level_name:
            proj = "-300"
        else:
            proj = level_name

        return f"{bucket} {str(row.get('SetLabel', period)).strip()} {proj}".strip()

    # POC labels
    if period_type == "weekly":
        bucket = "W"
    elif period_type == "monthly":
        bucket = "M"
    elif period_type == "yearly":
        bucket = "Y"
    else:
        bucket = "P"

    return f"{bucket} {period}".strip()


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
    chart_end = chart_df.index.max()
    for _, row in levels.iterrows():
        style = STYLE_MAP.get(str(row["PeriodType"]), {"label": row["PeriodType"], "color": "#CBD5E1", "width": 1.8, "dash": "solid"})
        anchor_col = "AnchorStart" if settings.extend_from_activation or row["Source"] == "IB" else "PeriodEnd"
        x0 = pd.Timestamp(row.get(anchor_col, row.get("PeriodEnd")))
        if pd.isna(x0):
            x0 = pd.Timestamp(row.get("PeriodEnd"))
        if x0 < chart_df.index.min():
            x0 = chart_df.index.min()
        x1 = pd.Timestamp(row["DisplayUntil"]) if pd.notna(row.get("DisplayUntil")) else chart_end
        if x1 > chart_end:
            x1 = chart_end
        if x1 < chart_df.index.min():
            continue
        opacity = 0.95 if bool(row["Valid"]) else 0.35
        name = style["label"]
        badge = format_level_badge(row)
        hover_text = (
            f"<b>{row['Source']} – {name}</b><br>"
            f"Badge: {badge}<br>"
            f"Level: {row['LevelName']}<br>"
            f"Period: {row['Period']}<br>"
            f"Price: {float(row['LevelPrice']):.2f}<br>"
            f"Distance: {float(row['SignedDist']):+.2f}<br>"
            f"Status: {row['Status']}<br>"
            f"<extra></extra>"
        )
        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[row["LevelPrice"], row["LevelPrice"]],
            mode="lines",
            line=dict(color=style["color"], width=style["width"], dash=style["dash"]),
            opacity=opacity,
            name=name,
            legendgroup=str(row["PeriodType"]),
            showlegend=str(row["PeriodType"]) not in legend_done,
            hovertemplate=hover_text,
        ))
        legend_done.add(str(row["PeriodType"]))
        if settings.show_labels:
            fig.add_annotation(
                x=x1,
                y=row["LevelPrice"],
                text=badge,
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
        title=dict(text=f"<b>{ticker}</b> – Level dashboard ({settings.display_mode})", x=0.02),
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220",
        height=780,
        margin=dict(l=20, r=180 if settings.show_labels else 30, t=70, b=30),
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
    cols = [
        "Source", "PeriodType", "LevelFamily", "LevelName", "Period", "LevelPrice", "LastPrice",
        "SignedDist", "AbsDist", "Status", "TouchDate", "PeriodStart", "PeriodEnd", "AnchorStart"
    ]
    existing = [c for c in cols if c in df.columns]
    table = df[existing].copy()
    return table.rename(columns={
        "PeriodType": "Type",
        "LevelFamily": "Family",
        "LevelPrice": "Price",
        "LastPrice": "Last",
        "SignedDist": "Dist",
        "TouchDate": "TestedAt",
        "PeriodStart": "Start",
        "PeriodEnd": "End",
        "AnchorStart": "ActiveFrom",
    })


def metric_block(levels: pd.DataFrame) -> dict[str, str | int]:
    if levels.empty:
        return {"total": 0, "active": 0, "tested": 0, "nearest": "—", "poc": 0, "ib": 0}
    nearest = float(levels["SignedDist"].iloc[0]) if not levels.empty else None
    return {
        "total": int(len(levels)),
        "active": int(levels["Valid"].sum()),
        "tested": int((~levels["Valid"]).sum()),
        "nearest": "—" if nearest is None else f"{nearest:+.2f}",
        "poc": int((levels["Source"] == "POC").sum()),
        "ib": int((levels["Source"] == "IB").sum()),
    }


def resolve_source_files(source: DataSource, ticker: str) -> tuple[Path, Path]:
    raw_path = source.raw_data_dir / source.raw_file_pattern.format(ticker=ticker)
    poc_path = source.processed_data_dir / source.poc_file_pattern.format(ticker=ticker)
    return raw_path, poc_path


def build_ib_settings_from_sidebar() -> dict:
    st.sidebar.markdown("---")
    st.sidebar.subheader("IB výpočet")
    monthly_enabled = st.sidebar.checkbox("Monthly IB", value=True)
    yearly_enabled = st.sidebar.checkbox("Yearly IB", value=True)
    standard_projection_enabled = st.sidebar.checkbox("Standard projekce", value=True)
    fibonacci_projection_enabled = st.sidebar.checkbox("Fib projekce", value=False)
    standard_multipliers = st.sidebar.multiselect(
        "Standard multipliers",
        options=[1.0, 1.5, 2.0, 3.0],
        default=[1.0, 1.5, 2.0, 3.0],
    )
    fibonacci_multipliers = st.sidebar.multiselect(
        "Fib multipliers",
        options=[0.618, 1.0, 1.272, 1.618, 2.618],
        default=[0.618, 1.0, 1.272, 1.618, 2.618],
    )
    return {
        "enabled": True,
        "yearly_enabled": yearly_enabled,
        "monthly_enabled": monthly_enabled,
        "monthly_mode": "first_5_trading_days",
        "hold_until_tested": True,
        "standard_projection_enabled": standard_projection_enabled,
        "standard_multipliers": standard_multipliers or [1.0, 1.5, 2.0, 3.0],
        "fibonacci_projection_enabled": fibonacci_projection_enabled,
        "fibonacci_multipliers": fibonacci_multipliers or [0.618, 1.0, 1.272, 1.618, 2.618],
    }


def sidebar_controls(sources: dict[str, DataSource], default_source: str, tickers: Iterable[str]) -> tuple[str, str, ChartSettings, dict]:
    st.sidebar.header("Nastavení")
    source_names = list(sources.keys())
    source_name = st.sidebar.selectbox("Zdroj dat", options=source_names, index=source_names.index(default_source))
    ticker = st.sidebar.selectbox("Ticker", options=list(tickers))
    display_mode = st.sidebar.radio("Režim zobrazení", DISPLAY_MODE_OPTIONS, index=2)

    selected_poc_periods = st.sidebar.multiselect("POC periody", POC_PERIODS, default=["monthly", "yearly"])
    selected_ib_periods = st.sidebar.multiselect(
        "IB typy",
        IB_PERIODS,
        default=["monthly_ib", "yearly_ib", "monthly_ib_std", "yearly_ib_std"],
    )
    selected_ib_families = st.sidebar.multiselect(
        "IB rodiny",
        IB_FAMILIES,
        default=["ib_core", "ib_standard"],
    )
    available_years = list(range(2000, 2036))
    selected_years = st.sidebar.multiselect("Rok sady", available_years, default=available_years)
    month_options = list(range(1, 13))
    selected_months = st.sidebar.multiselect("Měsíc sady", month_options, default=month_options)
    level_status = st.sidebar.selectbox("Stav levelu", ["All", "Only active", "Only tested"], index=0)
    nearest_only = st.sidebar.checkbox("Zobrazit jen nejbližší levely", value=True)
    nearest_count = st.sidebar.slider("Počet nejbližších levelů", 1, 20, 8)
    months_back = st.sidebar.slider("Kolik měsíců historie v grafu", 3, 48, 12)
    show_labels = st.sidebar.checkbox("Popisky levelů vpravo", value=True)
    extend_from_activation = st.sidebar.checkbox("Vést čáru od aktivace / konce periody", value=True)
    show_confluence_only = st.sidebar.checkbox("V režimu POC + IB ukázat jen konfluence", value=False)
    confluence_max_atr = st.sidebar.slider("Confluence šířka (proxy ATR)", 0.05, 1.00, 0.25, 0.05)

    ib_settings = build_ib_settings_from_sidebar()

    settings = ChartSettings(
        display_mode=display_mode,
        selected_poc_periods=selected_poc_periods,
        selected_ib_periods=selected_ib_periods,
        selected_years=selected_years,
        selected_months=selected_months,
        level_status=level_status,
        show_labels=show_labels,
        nearest_only=nearest_only,
        nearest_count=nearest_count,
        months_back=months_back,
        extend_from_activation=extend_from_activation,
        show_confluence_only=show_confluence_only,
        confluence_max_atr=confluence_max_atr,
    )
    return source_name, ticker, settings, ib_settings


def main() -> None:
    st.set_page_config(page_title="Level Dashboard", layout="wide")
    st.title("Level Dashboard")
    st.caption("Vizuální kontrola POC a IB levelů nad lokálními daty.")

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

    source_name, ticker, chart_settings, ib_settings = sidebar_controls(sources, default_source, tickers)
    source = sources[source_name]
    raw_path, poc_path = resolve_source_files(source, ticker)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Raw: `{raw_path}`")
    st.sidebar.caption(f"POC: `{poc_path}`")

    try:
        ohlcv = load_ohlcv_from_csv(str(raw_path))
    except Exception as exc:
        st.error(f"Nepodařilo se načíst raw data pro {ticker}: {exc}")
        st.stop()

    poc_levels = pd.DataFrame()
    try:
        poc_levels = load_poc_data(str(poc_path), ticker_hint=ticker)
    except Exception as exc:
        st.warning(f"POC data se nepodařilo načíst: {exc}")

    ib_levels = pd.DataFrame()
    try:
        ib_levels = compute_ib_levels_from_ohlcv(str(raw_path), ticker=ticker, ib_settings_dict=ib_settings)
    except Exception as exc:
        st.warning(f"IB data se nepodařilo spočítat: {exc}")

    prepared_frames: list[pd.DataFrame] = []
    ticker_poc = poc_levels[poc_levels.get("Ticker", "").astype(str).str.upper() == ticker].copy() if not poc_levels.empty else pd.DataFrame()
    if not ticker_poc.empty:
        prepared_frames.append(prepare_poc_levels(ticker_poc, ohlcv))
    if not ib_levels.empty:
        prepared_frames.append(prepare_ib_levels(ib_levels, ohlcv))

    if not prepared_frames:
        st.error("Nepodařilo se připravit žádné POC ani IB levely pro zobrazení.")
        st.stop()

    unified = pd.concat(prepared_frames, ignore_index=True)
    filtered = filter_unified_levels(unified, chart_settings)

    metrics = metric_block(filtered)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Zobrazené levely", metrics["total"])
    c2.metric("Aktivní", metrics["active"])
    c3.metric("Testované", metrics["tested"])
    c4.metric("POC", metrics["poc"])
    c5.metric("IB", metrics["ib"])
    c6.metric("Nejbližší distance", metrics["nearest"])

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
            "Price": st.column_config.NumberColumn(format="%.2f"),
            "Last": st.column_config.NumberColumn(format="%.2f"),
            "Dist": st.column_config.NumberColumn(format="%+.2f"),
            "AbsDist": st.column_config.NumberColumn(format="%.2f"),
            "TestedAt": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Start": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "End": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "ActiveFrom": st.column_config.DateColumn(format="YYYY-MM-DD"),
        },
    )

    st.download_button(
        "Stáhnout aktuálně filtrovanou tabulku",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_levels_filtered.csv",
        mime="text/csv",
    )

    with st.expander("Jak číst dashboard"):
        st.write(
            "POC levely se berou z processed POC CSV. IB levely se počítají přímo z raw OHLCV dat. "
            "V režimu POC + IB se zobrazují obě vrstvy zároveň. Aktivní IB level končí až při prvním testu, "
            "takže můžeš vizuálně ověřit, jestli životnost a projekce sedí." 
        )


if __name__ == "__main__":
    main()
