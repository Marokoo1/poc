from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

TRADES_FILE = PROCESSED_DIR / "poc_backtest_trades.csv"
SUMMARY_FILE = PROCESSED_DIR / "poc_backtest_summary.csv"

OHLCV_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


@st.cache_data(show_spinner=False, ttl=60)
def load_trades() -> pd.DataFrame:
    if not TRADES_FILE.exists():
        raise FileNotFoundError(f"Soubor neexistuje: {TRADES_FILE}")

    df = pd.read_csv(TRADES_FILE)

    date_cols = ["active_from", "touch_date", "entry_date", "exit_date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


@st.cache_data(show_spinner=False, ttl=60)
def load_summary() -> pd.DataFrame:
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(f"Soubor neexistuje: {SUMMARY_FILE}")

    return pd.read_csv(SUMMARY_FILE)


@st.cache_data(show_spinner=False, ttl=60)
def load_ohlcv(ticker: str) -> pd.DataFrame:
    path = RAW_DIR / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Raw data soubor neexistuje: {path}")

    df = pd.read_csv(path)
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} nemá požadované sloupce: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=OHLCV_COLUMNS).sort_values("Date").reset_index(drop=True)
    return df


def build_trade_chart(ohlcv: pd.DataFrame, trade: pd.Series, history_bars: int = 40) -> go.Figure:
    entry_date = pd.Timestamp(trade["entry_date"]) if pd.notna(trade["entry_date"]) else None
    exit_date = pd.Timestamp(trade["exit_date"]) if pd.notna(trade["exit_date"]) else None
    touch_date = pd.Timestamp(trade["touch_date"]) if pd.notna(trade["touch_date"]) else None

    if entry_date is None:
        raise ValueError("Vybraný obchod nemá entry_date.")

    entry_idx_list = ohlcv.index[ohlcv["Date"] == entry_date].tolist()
    entry_idx = entry_idx_list[0] if entry_idx_list else max(0, len(ohlcv) - 100)

    start_idx = max(0, entry_idx - history_bars)

    if exit_date is not None:
        exit_idx_list = ohlcv.index[ohlcv["Date"] == exit_date].tolist()
        exit_idx = exit_idx_list[0] if exit_idx_list else min(len(ohlcv) - 1, entry_idx + 30)
        end_idx = min(len(ohlcv) - 1, exit_idx + 20)
    else:
        end_idx = min(len(ohlcv) - 1, entry_idx + 30)

    chart_df = ohlcv.iloc[start_idx : end_idx + 1].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=chart_df["Date"],
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name="Cena",
            increasing_line_color="#22C55E",
            decreasing_line_color="#EF4444",
            whiskerwidth=0.35,
            showlegend=False,
        )
    )

    level_price = float(trade["level_price"])
    entry_price = float(trade["entry_price"]) if pd.notna(trade["entry_price"]) else None
    exit_price = float(trade["exit_price"]) if pd.notna(trade["exit_price"]) else None
    stop_price = float(trade["stop_price"]) if pd.notna(trade["stop_price"]) else None
    target_price = float(trade["target_price"]) if pd.notna(trade["target_price"]) else None

    x_min = chart_df["Date"].min()
    x_max = chart_df["Date"].max()

    fig.add_trace(
        go.Scatter(
            x=[x_min, x_max],
            y=[level_price, level_price],
            mode="lines",
            name="Level",
            line=dict(color="#F59E0B", width=2, dash="dash"),
            hovertemplate=f"Level: {level_price:.2f}<extra></extra>",
        )
    )

    if stop_price is not None:
        fig.add_trace(
            go.Scatter(
                x=[entry_date, x_max],
                y=[stop_price, stop_price],
                mode="lines",
                name="Stop",
                line=dict(color="#EF4444", width=1.5, dash="dot"),
                hovertemplate=f"Stop: {stop_price:.2f}<extra></extra>",
            )
        )

    if target_price is not None:
        fig.add_trace(
            go.Scatter(
                x=[entry_date, x_max],
                y=[target_price, target_price],
                mode="lines",
                name="Target",
                line=dict(color="#22C55E", width=1.5, dash="dot"),
                hovertemplate=f"Target: {target_price:.2f}<extra></extra>",
            )
        )

    if touch_date is not None:
        fig.add_trace(
            go.Scatter(
                x=[touch_date],
                y=[level_price],
                mode="markers",
                name="Touch",
                marker=dict(size=11, color="#F59E0B", symbol="diamond"),
                hovertemplate=f"Touch: {touch_date.date()}<extra></extra>",
            )
        )

    if entry_price is not None:
        fig.add_trace(
            go.Scatter(
                x=[entry_date],
                y=[entry_price],
                mode="markers",
                name="Entry",
                marker=dict(size=12, color="#60A5FA", symbol="triangle-up"),
                hovertemplate=f"Entry: {entry_price:.2f}<extra></extra>",
            )
        )

    if exit_date is not None and exit_price is not None:
        fig.add_trace(
            go.Scatter(
                x=[exit_date],
                y=[exit_price],
                mode="markers",
                name="Exit",
                marker=dict(size=12, color="#E5E7EB", symbol="x"),
                hovertemplate=f"Exit: {exit_price:.2f}<extra></extra>",
            )
        )

    title = (
        f"{trade['ticker']} | {trade['period_type']} | {trade['period']} | "
        f"{trade['side']} | {trade['exit_reason']}"
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220",
        height=760,
        margin=dict(l=20, r=20, t=60, b=20),
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


def build_overview_metrics(trades: pd.DataFrame) -> dict:
    real = trades[trades["entry_date"].notna()].copy()
    wins = real[real["pnl_abs"] > 0].copy()

    total_levels = len(trades)
    total_trades = len(real)
    win_rate = (len(wins) / total_trades * 100) if total_trades else 0.0
    avg_pnl_atr = real["pnl_atr"].mean() if "pnl_atr" in real.columns else float("nan")
    avg_hold = real["bars_held"].mean() if "bars_held" in real.columns else float("nan")

    return {
        "total_levels": total_levels,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_pnl_atr": avg_pnl_atr,
        "avg_hold": avg_hold,
    }


def build_ticker_summary(trades: pd.DataFrame) -> pd.DataFrame:
    real = trades[trades["entry_date"].notna()].copy()
    if real.empty:
        return pd.DataFrame()

    real["win"] = real["pnl_abs"] > 0

    out = (
        real.groupby("ticker", observed=False)
        .agg(
            trades=("ticker", "count"),
            win_rate=("win", "mean"),
            avg_pnl_abs=("pnl_abs", "mean"),
            avg_pnl_atr=("pnl_atr", "mean"),
            avg_hold=("bars_held", "mean"),
        )
        .reset_index()
        .sort_values(["avg_pnl_atr", "win_rate"], ascending=[False, False])
    )

    out["win_rate"] = (out["win_rate"] * 100).round(2)
    return out
    
def build_equity_df(trades: pd.DataFrame) -> pd.DataFrame:
    real = trades[trades["entry_date"].notna()].copy()
    if real.empty:
        return pd.DataFrame()

    real["exit_date"] = pd.to_datetime(real["exit_date"], errors="coerce")
    real = real.dropna(subset=["exit_date", "pnl_abs"]).copy()

    if real.empty:
        return pd.DataFrame()

    real = real.sort_values(["exit_date", "ticker", "period_type"]).reset_index(drop=True)
    real["equity"] = real["pnl_abs"].cumsum()
    real["running_max"] = real["equity"].cummax()
    real["drawdown"] = real["equity"] - real["running_max"]
    return real


def build_equity_by_period(trades: pd.DataFrame) -> pd.DataFrame:
    real = trades[trades["entry_date"].notna()].copy()
    if real.empty:
        return pd.DataFrame()

    real["exit_date"] = pd.to_datetime(real["exit_date"], errors="coerce")
    real = real.dropna(subset=["exit_date", "pnl_abs"]).copy()

    if real.empty:
        return pd.DataFrame()

    real = real.sort_values(["period_type", "exit_date", "ticker"]).reset_index(drop=True)
    real["equity_by_period"] = real.groupby("period_type", observed=False)["pnl_abs"].cumsum()
    return real


def build_performance_metrics(trades: pd.DataFrame) -> dict:
    eq = build_equity_df(trades)
    if eq.empty:
        return {
            "total_pnl": 0.0,
            "profit_factor": float("nan"),
            "max_drawdown": float("nan"),
            "avg_trade": float("nan"),
            "median_trade": float("nan"),
        }

    gross_profit = eq.loc[eq["pnl_abs"] > 0, "pnl_abs"].sum()
    gross_loss = -eq.loc[eq["pnl_abs"] < 0, "pnl_abs"].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "total_pnl": float(eq["pnl_abs"].sum()),
        "profit_factor": float(profit_factor),
        "max_drawdown": float(eq["drawdown"].min()),
        "avg_trade": float(eq["pnl_abs"].mean()),
        "median_trade": float(eq["pnl_abs"].median()),
    }


def plot_equity_curve(equity_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=equity_df["exit_date"],
            y=equity_df["equity"],
            mode="lines",
            name="Equity",
            line=dict(width=2),
        )
    )

    fig.update_layout(
        title="Equity curve",
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220",
        height=460,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Exit date",
        yaxis_title="Kumulativní PnL",
        hovermode="x unified",
    )
    return fig


def plot_drawdown_curve(equity_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=equity_df["exit_date"],
            y=equity_df["drawdown"],
            mode="lines",
            name="Drawdown",
            line=dict(width=2),
            fill="tozeroy",
        )
    )

    fig.update_layout(
        title="Drawdown",
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Exit date",
        yaxis_title="Drawdown",
        hovermode="x unified",
    )
    return fig


def plot_equity_by_period(equity_period_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    for period_type in sorted(equity_period_df["period_type"].dropna().unique().tolist()):
        subset = equity_period_df[equity_period_df["period_type"] == period_type].copy()
        subset = subset.sort_values("exit_date")

        fig.add_trace(
            go.Scatter(
                x=subset["exit_date"],
                y=subset["equity_by_period"],
                mode="lines",
                name=str(period_type),
                line=dict(width=2),
            )
        )

    fig.update_layout(
        title="Equity podle typu levelu",
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220",
        height=460,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Exit date",
        yaxis_title="Kumulativní PnL",
        hovermode="x unified",
    )
    return fig
    
def filter_trades_for_analysis(
    trades: pd.DataFrame,
    ticker: str = "ALL",
    period_type: str = "ALL",
    side: str = "ALL",
    exit_reason: str = "ALL",
    trend_aligned: str = "ALL",
    year: str = "ALL",
    pnl_mode: str = "ALL",
    hold_range: tuple[int, int] | None = None,
    pnl_atr_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    df = trades.copy()

    if ticker != "ALL":
        df = df[df["ticker"] == ticker]

    if period_type != "ALL":
        df = df[df["period_type"] == period_type]

    if side != "ALL":
        df = df[df["side"] == side]

    if exit_reason != "ALL":
        df = df[df["exit_reason"] == exit_reason]

    if trend_aligned != "ALL":
        df = df[df["trend_aligned"].astype(str) == trend_aligned]

    if year != "ALL" and "entry_date" in df.columns:
        df = df[
            df["entry_date"].dt.year.astype("Int64").astype(str) == year
        ]

    if pnl_mode == "Winners only":
        df = df[df["pnl_abs"] > 0]
    elif pnl_mode == "Losers only":
        df = df[df["pnl_abs"] <= 0]

    if hold_range is not None and "bars_held" in df.columns:
        df = df[
            df["bars_held"].fillna(-1).between(hold_range[0], hold_range[1])
        ]

    if pnl_atr_range is not None and "pnl_atr" in df.columns:
        df = df[
            df["pnl_atr"].fillna(0).between(pnl_atr_range[0], pnl_atr_range[1])
        ]

    return df

def build_summary_by_level_type(trades: pd.DataFrame) -> pd.DataFrame:
    real = trades[trades["entry_date"].notna()].copy()
    if real.empty:
        return pd.DataFrame()

    real["win"] = real["pnl_abs"] > 0

    out = (
        real.groupby("period_type", observed=False)
        .agg(
            trades=("period_type", "count"),
            win_rate=("win", "mean"),
            avg_pnl_abs=("pnl_abs", "mean"),
            avg_pnl_atr=("pnl_atr", "mean"),
            avg_hold=("bars_held", "mean"),
        )
        .reset_index()
        .sort_values("period_type")
    )

    out["win_rate"] = (out["win_rate"] * 100).round(2)
    return out


def build_active_filter_caption(
    ticker: str,
    period_type: str,
    side: str,
    exit_reason: str,
    trend_aligned: str,
    year: str,
    pnl_mode: str,
    hold_range: tuple[int, int] | None,
    pnl_atr_range: tuple[float, float] | None,
) -> str:
    parts = []

    if ticker != "ALL":
        parts.append(f"Ticker: {ticker}")
    if period_type != "ALL":
        parts.append(f"Period: {period_type}")
    if side != "ALL":
        parts.append(f"Side: {side}")
    if exit_reason != "ALL":
        parts.append(f"Exit: {exit_reason}")
    if trend_aligned != "ALL":
        parts.append(f"Trend aligned: {trend_aligned}")
    if year != "ALL":
        parts.append(f"Year: {year}")
    if pnl_mode != "ALL":
        parts.append(f"PnL: {pnl_mode}")
    if hold_range is not None:
        parts.append(f"Hold: {hold_range[0]}–{hold_range[1]}")
    if pnl_atr_range is not None:
        parts.append(f"PnL ATR: {pnl_atr_range[0]:.2f} až {pnl_atr_range[1]:.2f}")

    return " | ".join(parts) if parts else "Bez filtru"

def main() -> None:
    st.set_page_config(page_title="POC Backtest Dashboard", layout="wide")
    st.title("POC Backtest Dashboard")

    try:
        trades = load_trades()
        _ = load_summary()
    except Exception as e:
        st.error(str(e))
        st.stop()

    real_trades = trades[trades["entry_date"].notna()].copy()

    if real_trades.empty:
        st.warning("Nejsou k dispozici žádné obchody s entry.")
        st.stop()

    st.subheader("Přehled backtestu a vizualizace jednotlivých obchodů")

    # =========================
    # GLOBÁLNÍ FILTRY
    # =========================
    st.markdown("### Globální filtry")

    f1, f2, f3, f4, f5, f6 = st.columns(6)

    tickers = ["ALL"] + sorted(real_trades["ticker"].dropna().unique().tolist())
    period_types = ["ALL"] + sorted(real_trades["period_type"].dropna().unique().tolist())
    sides = ["ALL"] + sorted(real_trades["side"].dropna().unique().tolist())
    exit_reasons = ["ALL"] + sorted(real_trades["exit_reason"].dropna().unique().tolist())
    trend_flags = ["ALL", "True", "False"]

    year_values = []
    if "entry_date" in real_trades.columns:
        year_values = sorted(real_trades["entry_date"].dropna().dt.year.astype(int).unique().tolist())
    years = ["ALL"] + [str(y) for y in year_values]

    global_ticker = f1.selectbox("Ticker", tickers, key="global_ticker")
    global_period = f2.selectbox("Period type", period_types, key="global_period")
    global_side = f3.selectbox("Side", sides, key="global_side")
    global_exit = f4.selectbox("Exit reason", exit_reasons, key="global_exit")
    global_trend = f5.selectbox("Trend aligned", trend_flags, key="global_trend")
    global_year = f6.selectbox("Year", years, key="global_year")

    g1, g2, g3 = st.columns(3)

    global_pnl_mode = g1.selectbox(
        "PnL filter",
        ["ALL", "Winners only", "Losers only"],
        key="global_pnl_mode",
    )

    hold_min, hold_max = 0, 200
    if "bars_held" in real_trades.columns and real_trades["bars_held"].notna().any():
        hold_min = int(real_trades["bars_held"].min())
        hold_max = int(real_trades["bars_held"].max())
        if hold_min > hold_max:
            hold_min, hold_max = 0, 200

    global_hold_range = g2.slider(
        "Hold bars range",
        min_value=hold_min,
        max_value=hold_max,
        value=(hold_min, hold_max),
        key="global_hold_range",
    )

    pnl_atr_series = real_trades["pnl_atr"].dropna() if "pnl_atr" in real_trades.columns else pd.Series(dtype=float)
    if not pnl_atr_series.empty:
        pnl_atr_min = float(pnl_atr_series.min())
        pnl_atr_max = float(pnl_atr_series.max())
    else:
        pnl_atr_min, pnl_atr_max = -5.0, 5.0

    global_pnl_atr_range = g3.slider(
        "PnL ATR range",
        min_value=float(pnl_atr_min),
        max_value=float(pnl_atr_max),
        value=(float(pnl_atr_min), float(pnl_atr_max)),
        key="global_pnl_atr_range",
    )

    filtered_trades = filter_trades_for_analysis(
        real_trades,
        ticker=global_ticker,
        period_type=global_period,
        side=global_side,
        exit_reason=global_exit,
        trend_aligned=global_trend,
        year=global_year,
        pnl_mode=global_pnl_mode,
        hold_range=global_hold_range,
        pnl_atr_range=global_pnl_atr_range,
    ).copy()

    filtered_trades = filtered_trades.sort_values(
        ["entry_date", "ticker"], ascending=[False, True]
    ).reset_index(drop=True)

    active_filter_caption = build_active_filter_caption(
        ticker=global_ticker,
        period_type=global_period,
        side=global_side,
        exit_reason=global_exit,
        trend_aligned=global_trend,
        year=global_year,
        pnl_mode=global_pnl_mode,
        hold_range=global_hold_range,
        pnl_atr_range=global_pnl_atr_range,
    )

    st.caption(f"Aktivní filtr: {active_filter_caption}")

    # =========================
    # HORNÍ METRIKY Z FILTRU
    # =========================
    metrics = build_overview_metrics(filtered_trades)
    perf = build_performance_metrics(filtered_trades)
    equity_df = build_equity_df(filtered_trades)
    equity_period_df = build_equity_by_period(filtered_trades)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Levelů", f"{metrics['total_levels']:,}")
    col2.metric("Obchodů", f"{metrics['total_trades']:,}")
    col3.metric("Win rate", f"{metrics['win_rate']:.2f}%")
    col4.metric("Avg PnL ATR", f"{metrics['avg_pnl_atr']:.3f}" if pd.notna(metrics["avg_pnl_atr"]) else "n/a")
    col5.metric("Avg hold", f"{metrics['avg_hold']:.2f} d" if pd.notna(metrics["avg_hold"]) else "n/a")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total PnL", f"{perf['total_pnl']:.2f}")
    m2.metric("Profit factor", f"{perf['profit_factor']:.3f}" if pd.notna(perf["profit_factor"]) else "n/a")
    m3.metric("Max drawdown", f"{perf['max_drawdown']:.2f}" if pd.notna(perf["max_drawdown"]) else "n/a")
    m4.metric("Avg trade", f"{perf['avg_trade']:.3f}" if pd.notna(perf["avg_trade"]) else "n/a")
    m5.metric("Median trade", f"{perf['median_trade']:.3f}" if pd.notna(perf["median_trade"]) else "n/a")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Summary", "Ticker summary", "Trades explorer", "Trade chart", "Equity"]
    )

    # =========================
    # TAB 1
    # =========================
    with tab1:
        st.subheader("Souhrn podle typu levelu")
        summary_filtered = build_summary_by_level_type(filtered_trades)

        if summary_filtered.empty:
            st.info("Žádné obchody neodpovídají zvoleným filtrům.")
        else:
            st.dataframe(summary_filtered, width="stretch", hide_index=True)

    # =========================
    # TAB 2
    # =========================
    with tab2:
        st.subheader("Souhrn podle tickeru")
        ticker_summary = build_ticker_summary(filtered_trades)

        if ticker_summary.empty:
            st.info("Žádné obchody neodpovídají zvoleným filtrům.")
        else:
            st.dataframe(ticker_summary, width="stretch", hide_index=True)

    # =========================
    # TAB 3
    # =========================
    with tab3:
        st.subheader("Trades explorer")
        st.write(f"Nalezeno obchodů: **{len(filtered_trades)}**")
        st.dataframe(filtered_trades, width="stretch", hide_index=True)

    # =========================
    # TAB 4
    # =========================
    with tab4:
        st.subheader("Graf konkrétního obchodu")

        if filtered_trades.empty:
            st.warning("Žádné obchody neodpovídají zvoleným filtrům.")
        else:
            chart_filtered = filtered_trades.copy()

            g1, g2, g3 = st.columns(3)

            sort_by = g1.selectbox(
                "Sort by",
                ["entry_date", "pnl_abs", "pnl_atr", "bars_held", "ticker"],
                key="chart_sort_by",
            )

            sort_order = g2.selectbox(
                "Order",
                ["Descending", "Ascending"],
                key="chart_sort_order",
            )

            history_bars = int(
                g3.slider(
                    "History bars",
                    min_value=20,
                    max_value=120,
                    value=40,
                    step=10,
                    key="chart_history_bars",
                )
            )

            ascending = sort_order == "Ascending"
            chart_filtered = chart_filtered.sort_values(
                [sort_by, "ticker"],
                ascending=[ascending, True],
                na_position="last",
            ).reset_index(drop=True)

            st.write(f"Nalezeno obchodů pro graf: **{len(chart_filtered)}**")

            chart_filtered["label"] = (
                chart_filtered["ticker"].astype(str)
                + " | "
                + chart_filtered["period_type"].astype(str)
                + " | "
                + chart_filtered["period"].astype(str)
                + " | "
                + chart_filtered["side"].astype(str)
                + " | "
                + chart_filtered["entry_date"].dt.strftime("%Y-%m-%d")
                + " | "
                + chart_filtered["exit_reason"].astype(str)
                + " | pnl="
                + chart_filtered["pnl_abs"].round(2).astype(str)
                + " | atr="
                + chart_filtered["pnl_atr"].round(2).astype(str)
            )

            selected_label = st.selectbox(
                "Vyber obchod",
                chart_filtered["label"].tolist(),
                key="chart_trade_select",
            )

            selected_trade = chart_filtered.loc[
                chart_filtered["label"] == selected_label
            ].iloc[0]

            ticker = str(selected_trade["ticker"])
            ohlcv = load_ohlcv(ticker)

            st.plotly_chart(
                build_trade_chart(ohlcv, selected_trade, history_bars=history_bars),
                width="stretch",
            )

            info_cols = st.columns(8)
            info_cols[0].metric("Ticker", ticker)
            info_cols[1].metric("Period", str(selected_trade["period_type"]))
            info_cols[2].metric("Side", str(selected_trade["side"]))
            info_cols[3].metric("Exit", str(selected_trade["exit_reason"]))
            info_cols[4].metric("PnL", f"{selected_trade['pnl_abs']:.2f}")
            info_cols[5].metric(
                "PnL ATR",
                f"{selected_trade['pnl_atr']:.2f}" if pd.notna(selected_trade["pnl_atr"]) else "n/a"
            )
            info_cols[6].metric(
                "Hold",
                f"{selected_trade['bars_held']:.0f} d" if pd.notna(selected_trade["bars_held"]) else "n/a"
            )
            info_cols[7].metric("Trend aligned", str(selected_trade["trend_aligned"]))

            detail = pd.DataFrame([selected_trade.drop(labels=["label"])])
            st.dataframe(detail, width="stretch", hide_index=True)

    # =========================
    # TAB 5
    # =========================
    with tab5:
        st.subheader("Equity a drawdown")

        st.write(f"Obchodů pro equity: **{len(equity_df)}**")

        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Total PnL", f"{perf['total_pnl']:.2f}")
        p2.metric("Profit factor", f"{perf['profit_factor']:.3f}" if pd.notna(perf["profit_factor"]) else "n/a")
        p3.metric("Max drawdown", f"{perf['max_drawdown']:.2f}" if pd.notna(perf["max_drawdown"]) else "n/a")
        p4.metric("Avg trade", f"{perf['avg_trade']:.3f}" if pd.notna(perf["avg_trade"]) else "n/a")
        p5.metric("Median trade", f"{perf['median_trade']:.3f}" if pd.notna(perf["median_trade"]) else "n/a")

        if equity_df.empty:
            st.warning("Žádné uzavřené obchody neodpovídají zvoleným filtrům.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_equity_curve(equity_df), width="stretch")
            with c2:
                st.plotly_chart(plot_drawdown_curve(equity_df), width="stretch")

            if not equity_period_df.empty:
                st.plotly_chart(plot_equity_by_period(equity_period_df), width="stretch")

            preview_cols = [
                "ticker",
                "period_type",
                "side",
                "entry_date",
                "exit_date",
                "exit_reason",
                "pnl_abs",
                "equity",
                "drawdown",
            ]
            preview_cols = [c for c in preview_cols if c in equity_df.columns]
            st.dataframe(equity_df[preview_cols], width="stretch", hide_index=True)

if __name__ == "__main__":
    main()
