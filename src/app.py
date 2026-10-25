import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, time
from zoneinfo import ZoneInfo

from src.stock_search import search_stocks
from src.next_day_predictor import train_and_predict_next_close


# --------------------------------------------------
# 1. PAGE CONFIGURATION AND CONSTANTS
# --------------------------------------------------
st.set_page_config(
    page_title="Stock Intelligence | Market Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PERIOD_OPTIONS = {
    "1 month": "1mo",
    "3 months": "3mo",
    "6 months": "6mo",
    "1 year": "1y",
    "5 years": "5y",
    "Maximum available": "max",
}

REQUIRED_HISTORY_COLUMNS = {"Open", "High", "Low", "Close"}


# --------------------------------------------------
# 2. SESSION STATE
# --------------------------------------------------
DEFAULT_STATE = {
    "search_results": [],
    "search_query_saved": "",
    "search_market_saved": "",
    "loaded_history": None,
    "loaded_ticker": "",
    "loaded_period": "",
    "loaded_market": "",
    "loaded_prediction_report": None,
    "loaded_prediction_error": "",
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def clear_loaded_analysis():
    """Clear the previously selected ticker's data and forecast."""
    defaults = {
        "loaded_history": None,
        "loaded_ticker": "",
        "loaded_period": "",
        "loaded_market": "",
        "loaded_prediction_report": None,
        "loaded_prediction_error": "",
    }

    for key, value in defaults.items():
        st.session_state[key] = value


# --------------------------------------------------
# 3. DATA AND DISPLAY HELPERS
# --------------------------------------------------
def normalize_history_columns(history):
    """Flatten Yahoo Finance MultiIndex columns if necessary."""
    if not isinstance(history.columns, pd.MultiIndex):
        return history

    standard = {
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Dividends",
        "Stock Splits",
    }

    normalized = []

    for column in history.columns:
        parts = list(column)
        matches = [
            str(part)
            for part in parts
            if str(part) in standard
        ]

        normalized.append(
            matches[0]
            if matches
            else "_".join(map(str, parts))
        )

    history = history.copy()
    history.columns = normalized

    return history


def get_completed_daily_history(history, market):
    """Exclude today's daily bar if the scheduled market close has not passed."""
    if history.empty:
        return history

    timezone_name = (
        "Asia/Kolkata"
        if market == "India"
        else "America/New_York"
    )

    market_tz = ZoneInfo(timezone_name)
    now_local = datetime.now(market_tz)

    market_close = (
        time(15, 30)
        if market == "India"
        else time(16, 0)
    )

    latest = pd.Timestamp(history.index[-1])

    if latest.tzinfo is None:
        latest_local = latest.tz_localize(market_tz)
    else:
        latest_local = latest.tz_convert(market_tz)

    if (
        latest_local.date() == now_local.date()
        and now_local.time() < market_close
    ):
        history = history.iloc[:-1]

    return history


def download_stock_history(ticker, market, period):
    """Download and validate daily OHLCV history."""
    try:
        history = yf.Ticker(ticker).history(
            period=period,
            interval="1d",
            auto_adjust=False,
            actions=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to download data for {ticker}: {exc}"
        ) from exc

    if history is None or history.empty:
        raise RuntimeError(
            f"No historical price data was returned for {ticker}. "
            "Check the listing or try another symbol."
        )

    print(f"[DEBUG] {ticker} | Requested period: {period}")
    print(f"[DEBUG] Rows returned by Yahoo Finance: {len(history)}")

    history = normalize_history_columns(history)

    missing = REQUIRED_HISTORY_COLUMNS - set(history.columns)

    if missing:
        raise RuntimeError(
            "Downloaded data is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if "Volume" not in history.columns:
        history["Volume"] = float("nan")

    history = history.dropna(subset=["Close"]).copy()
    history = history.sort_index()
    history = history.loc[
        ~history.index.duplicated(keep="last")
    ]

    print(f"[DEBUG] Rows after cleaning: {len(history)}")

    history = get_completed_daily_history(history, market)

    print(f"[DEBUG] Rows after completed-day filter: {len(history)}")

    if not history.empty:
        print(f"[DEBUG] First date: {history.index.min()}")
        print(f"[DEBUG] Last date: {history.index.max()}")

    if history.empty:
        raise RuntimeError(
            "No completed daily closing price is available yet. "
            "Try again later."
        )

    return history


def format_price(value, currency_symbol):
    """Format a price safely."""
    if value is None or pd.isna(value):
        return "N/A"

    return f"{currency_symbol}{float(value):,.2f}"


def format_percent(value):
    """Format a percentage safely."""
    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value):+.2f}%"


def get_display_date(index_value):
    """Convert a date-like value to a readable date."""
    return pd.Timestamp(index_value).strftime("%d %B %Y")


def sanitize_filename(ticker):
    """Make a ticker safe for use in a downloaded filename."""
    return (
        str(ticker)
        .replace(".", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def make_candidate_comparison(candidate_scores):
    """Prepare cross-validation scores for display."""
    if not candidate_scores:
        return pd.DataFrame()

    scores = pd.DataFrame(candidate_scores).rename(
        columns={
            "model": "Model",
            "name": "Model",
            "mae": "CV MAE",
            "rmse": "CV RMSE",
            "r2": "CV R²",
            "cv_mae": "CV MAE",
            "cv_rmse": "CV RMSE",
            "cv_r2": "CV R²",
            "cross_validation_mae": "CV MAE",
            "cross_validation_rmse": "CV RMSE",
            "cross_validation_r2": "CV R²",
        }
    )

    scores = scores.loc[
        :,
        ~scores.columns.duplicated(),
    ]

    scores = scores.drop(
        columns=[
            "fold_maes",
            "fold_rmses",
            "fold_r2s",
        ],
        errors="ignore",
    )

    preferred = [
        column
        for column in ["Model", "CV MAE", "CV RMSE", "CV R²"]
        if column in scores.columns
    ]

    if preferred:
        scores = scores[
            preferred
            + [
                column
                for column in scores.columns
                if column not in preferred
            ]
        ]

    return scores


# --------------------------------------------------
# 4. DARK TERMINAL THEME
# --------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --terminal-bg: #0b0f14;
        --terminal-panel: #111820;
        --terminal-border: #25313d;
        --terminal-text: #e6edf3;
        --terminal-muted: #91a0af;
        --terminal-accent: #35d49a;
    }

    .stApp {
        background: var(--terminal-bg);
        color: var(--terminal-text);
    }

    [data-testid="stHeader"] {
        background: rgba(11, 15, 20, .94);
    }

    [data-testid="stSidebar"] {
        background: var(--terminal-panel);
        border-right: 1px solid var(--terminal-border);
    }

    [data-testid="stSidebar"] > div {
        background: transparent;
    }

    [data-testid="stMetric"] {
        background: var(--terminal-panel);
        border: 1px solid var(--terminal-border);
        border-radius: 12px;
        padding: 15px 16px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--terminal-muted);
    }

    [data-testid="stMetricValue"] {
        color: var(--terminal-text);
    }

    h1, h2, h3 {
        color: var(--terminal-text);
        letter-spacing: -.025em;
    }

    [data-testid="stCaptionContainer"] {
        color: var(--terminal-muted);
    }

    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button[kind="primary"] {
        background: var(--terminal-accent);
        color: #07120e;
        border: 1px solid var(--terminal-accent);
        border-radius: 8px;
        font-weight: 700;
    }

    div.stButton > button[kind="primary"]:hover,
    div.stFormSubmitButton > button[kind="primary"]:hover {
        background: #63e6b5;
        border-color: #63e6b5;
        color: #07120e;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--terminal-border);
        border-radius: 10px;
        overflow: hidden;
    }

    hr {
        border-color: var(--terminal-border);
    }

    .terminal-eyebrow {
        color: #35d49a;
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .15em;
        text-transform: uppercase;
        margin-bottom: .35rem;
    }

    .terminal-subtitle {
        color: #91a0af;
        font-size: .96rem;
        line-height: 1.6;
    }

    .terminal-panel {
        background: #111820;
        border: 1px solid #25313d;
        border-radius: 14px;
        padding: 18px 20px;
    }

    .terminal-status {
        color: #35d49a;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .08em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 5. SIDEBAR CONTROLS
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding:8px 0 18px">
          <div style="font-size:.76rem;letter-spacing:.18em;
                      color:#35d49a;font-weight:800">
            MSR / MARKET LAB
          </div>
          <div style="font-size:1.35rem;font-weight:750;
                      color:#e6edf3;margin-top:7px">
            Stock Intelligence
          </div>
          <div style="font-size:.82rem;color:#91a0af;margin-top:5px">
            Research terminal · Daily data
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### MARKET")

    market_choice = st.radio(
        "Select market",
        options=["India", "United States"],
        key="market_selector",
    )

    market = (
        "India"
        if market_choice == "India"
        else "USA"
    )

    currency_symbol = (
        "₹"
        if market == "India"
        else "$"
    )

    if st.session_state["search_market_saved"] != market:
        st.session_state["search_results"] = []
        st.session_state["search_query_saved"] = ""

        clear_loaded_analysis()

        st.session_state["search_market_saved"] = market
        st.session_state.pop("selected_stock_symbol", None)

    st.divider()
    st.markdown("#### FIND AN INSTRUMENT")

    st.caption(
        "Examples: Reliance, TCS, Hindustan Copper, NIFTY 50."
        if market == "India"
        else "Examples: Apple, Microsoft, NVIDIA, AAPL."
    )

    with st.form("stock_search_form"):
        search_query = st.text_input(
            "Company or ticker",
            placeholder="e.g. HINDCOPPER.NS",
            help="Search by company name or exchange ticker.",
        ).strip()

        search_clicked = st.form_submit_button(
            "Search stocks",
            type="primary",
            width="stretch",
        )

    if search_clicked:
        st.session_state["search_results"] = []
        st.session_state["search_query_saved"] = ""

        clear_loaded_analysis()

        st.session_state.pop("selected_stock_symbol", None)

        if not search_query:
            st.warning("Enter a company name or ticker symbol.")
        else:
            try:
                with st.spinner(f"Searching for {search_query}..."):
                    found = search_stocks(
                        search_query,
                        market=market,
                    )

                if not found:
                    st.warning(
                        "No matching instruments found. Try another query."
                    )
                else:
                    matching = [
                        item
                        for item in found
                        if item.get("matches_market", False)
                    ]

                    displayed = matching or found

                    if not matching:
                        st.warning(
                            "The market could not be confidently verified. "
                            "Check the exchange before loading data."
                        )

                    st.session_state["search_results"] = displayed
                    st.session_state["search_query_saved"] = search_query

            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))

            except Exception as exc:
                st.error(f"Stock search failed: {exc}")

    results = st.session_state["search_results"]

    selected_symbol = None
    selected_stock = None
    period_label = None
    period = None
    load_clicked = False

    if results:
        result_by_symbol = {
            item["symbol"]: item
            for item in results
            if item.get("symbol")
        }

        symbols = list(result_by_symbol)

        if symbols:
            st.divider()
            st.markdown("#### LISTING")

            selected_symbol = st.selectbox(
                "Company / exchange",
                options=symbols,
                format_func=lambda symbol: (
                    f"{result_by_symbol[symbol].get('name', symbol)} "
                    f"· {symbol} "
                    f"({result_by_symbol[symbol].get('exchange', 'Unknown')})"
                ),
                key="selected_stock_symbol",
            )

            selected_stock = result_by_symbol[selected_symbol]

            if not selected_stock.get("matches_market", False):
                st.warning(
                    "Verify this listing belongs to the selected market."
                )

            st.divider()
            st.markdown("#### HISTORY WINDOW")

            period_label = st.selectbox(
                "Historical price range",
                options=list(PERIOD_OPTIONS),
                index=2,
                key="history_period_selector",
            )

            period = PERIOD_OPTIONS[period_label]

            load_clicked = st.button(
                "Analyze selected stock",
                type="primary",
                width="stretch",
            )

    else:
        st.caption(
            "Search for an instrument to choose its listing and history."
        )

    st.divider()

    st.markdown(
        """
        <div style="font-size:.76rem;color:#91a0af;line-height:1.6">
          DATA SOURCE<br>
          <span style="color:#e6edf3">Yahoo Finance via yfinance</span>
          <br><br>
          FORECAST HORIZON<br>
          <span style="color:#e6edf3">Next available trading session</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # --------------------------------------------------
# 6. DOWNLOAD HISTORY AND RUN FORECAST
# --------------------------------------------------
if load_clicked and selected_symbol and period:
    clear_loaded_analysis()

    try:
        with st.spinner(
            f"Downloading daily prices for {selected_symbol}..."
        ):
            history = download_stock_history(
                selected_symbol,
                market,
                period,
            )

        st.session_state["loaded_history"] = history
        st.session_state["loaded_ticker"] = selected_symbol
        st.session_state["loaded_period"] = period
        st.session_state["loaded_market"] = market

    except Exception as exc:
        st.error(f"Unable to load historical prices: {exc}")

    if st.session_state["loaded_history"] is not None:
        try:
            with st.spinner(
                f"Training and validating a next-session model for "
                f"{selected_symbol}..."
            ):
                # Always try to obtain up to five years for model training,
                # independently of the shorter chart display range.
                model_history = download_stock_history(
                    selected_symbol,
                    market,
                    "5y",
                )

                if len(model_history) < 150:
                    # Do not claim that a model has been trained when the
                    # listing has too little valid daily OHLC history.
                    latest = model_history.iloc[-1]
                    latest_close = float(latest["Close"])

                    latest_date = pd.Timestamp(model_history.index[-1])

                    if latest_date.tzinfo is not None:
                        latest_date = latest_date.tz_localize(None)

                    # This is the next weekday estimate, not an exchange-
                    # confirmed session; exchange holidays are not checked.
                    next_date = latest_date + pd.offsets.BDay(1)

                    report = {
                        "model_name": (
                            "Last-Close Baseline "
                            "(insufficient ML history)"
                        ),
                        "predicted_next_closing_price": latest_close,
                        "latest_closing_price": latest_close,
                        "expected_price_change": 0.0,
                        "expected_percentage_change": 0.0,
                        "latest_market_data_date": (
                            latest_date.strftime("%d %B %Y")
                        ),
                        "estimated_next_trading_date": (
                            next_date.strftime("%d %B %Y")
                        ),
                        "training_rows": 0,
                        "feature_count": 0,
                        "best_ml_model_name": None,
                        "ml_predicted_next_closing_price": None,
                        "ml_expected_price_change": None,
                        "ml_expected_percentage_change": None,
                        "ml_forecast_is_primary": False,
                        "forecast_mode": "baseline_only",
                        "available_ohlc_rows": int(len(model_history)),
                        "validation": {},
                    }

                else:
                    report = train_and_predict_next_close(model_history)

            st.session_state["loaded_prediction_report"] = report

        except Exception as exc:
            st.session_state["loaded_prediction_error"] = str(exc)


# --------------------------------------------------
# 7. CURRENT ANALYSIS CHECK
# --------------------------------------------------
loaded_history = st.session_state["loaded_history"]
loaded_ticker = st.session_state["loaded_ticker"]
loaded_period = st.session_state["loaded_period"]
loaded_market = st.session_state["loaded_market"]

current_period = PERIOD_OPTIONS.get(
    st.session_state.get("history_period_selector", ""),
    "",
)

current_results = st.session_state["search_results"]

current_result_by_symbol = {
    item["symbol"]: item
    for item in current_results
    if item.get("symbol")
}

selected_symbol_now = st.session_state.get("selected_stock_symbol")

analysis_is_current = (
    loaded_history is not None
    and bool(loaded_ticker)
    and loaded_ticker == selected_symbol_now
    and loaded_period == current_period
    and loaded_market == market
    and st.session_state["search_market_saved"] == market
    and loaded_ticker in current_result_by_symbol
)


# --------------------------------------------------
# 8. DASHBOARD HEADER
# --------------------------------------------------
st.markdown(
    '<div class="terminal-eyebrow">'
    'MARKET INTELLIGENCE / DAILY ANALYTICS</div>',
    unsafe_allow_html=True,
)

st.title("Stock Price Intelligence System")

st.markdown(
    """
    <div class="terminal-subtitle">
      Explore historical market behavior, compare forecast approaches,
      and inspect time-ordered model validation for Indian and US listings.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


if analysis_is_current:
    history = loaded_history.copy()
    latest_row = history.iloc[-1]
    latest_close = float(latest_row["Close"])
    latest_date = get_display_date(history.index[-1])

    selected_record = current_result_by_symbol[loaded_ticker]
    company_name = selected_record.get("name", loaded_ticker)

    previous_close = (
        float(history.iloc[-2]["Close"])
        if len(history) >= 2
        else None
    )

    if previous_close is not None and previous_close != 0:
        daily_change = latest_close - previous_close
        daily_change_pct = daily_change / previous_close * 100
    else:
        daily_change = None
        daily_change_pct = None

    title_col, status_col = st.columns([3, 1])

    with title_col:
        st.markdown(f"### {company_name}")

        st.caption(
            f"{loaded_ticker} · "
            f"{selected_record.get('exchange', 'Exchange unknown')} · "
            f"{selected_record.get('type', 'Instrument')} · "
            f"Latest completed daily bar: {latest_date}"
        )

    with status_col:
        st.markdown(
            '<div class="terminal-status">● ANALYSIS LOADED</div>',
            unsafe_allow_html=True,
        )

        st.caption(f"History window: {period_label or loaded_period}")

    overview_tab, forecast_tab, data_tab = st.tabs(
        [
            "Overview",
            "Forecast & Validation",
            "Historical Data",
        ]
    )

    # --------------------------------------------------
    # 9. OVERVIEW TAB
    # --------------------------------------------------
    with overview_tab:
        st.markdown("#### Market snapshot")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Latest close",
            format_price(latest_close, currency_symbol),
        )

        c2.metric(
            "Previous close",
            format_price(previous_close, currency_symbol),
        )

        c3.metric(
            "Daily change",
            format_price(daily_change, currency_symbol),
            (
                format_percent(daily_change_pct)
                if daily_change_pct is not None
                else None
            ),
        )

        c4.metric(
            "Trading volume",
            (
                f"{int(latest_row['Volume']):,}"
                if pd.notna(latest_row["Volume"])
                else "N/A"
            ),
            help="Volume may be unavailable for market indices.",
        )

        st.divider()
        st.markdown("#### Next-day closing-price forecast")

        overview_report = st.session_state.get(
            "loaded_prediction_report"
        )

        overview_error = st.session_state.get(
            "loaded_prediction_error",
            "",
        )

        if overview_report:
            if overview_report.get("forecast_mode") == "baseline_only":
                st.warning(
                    "Only "
                    f"{overview_report.get('available_ohlc_rows', 0)} "
                    "valid daily OHLC rows were available. A machine-learning "
                    "model cannot be trained reliably from this amount of "
                    "history. The value below is the last-close baseline, "
                    "not an ML prediction."
                )

            forecast_price = overview_report.get(
                "predicted_next_closing_price"
            )

            forecast_change = overview_report.get(
                "expected_price_change"
            )

            forecast_pct = overview_report.get(
                "expected_percentage_change"
            )

            forecast_date = overview_report.get(
                "estimated_next_trading_date",
                "Unknown",
            )

            forecast_model = overview_report.get(
                "model_name",
                "Unknown",
            )

            st.markdown(
                f"<div class='terminal-status'>"
                f"FORECAST READY · {forecast_model}</div>",
                unsafe_allow_html=True,
            )

            f1, f2, f3 = st.columns([1.5, 1, 1])

            f1.metric(
                "Estimated next closing price",
                format_price(forecast_price, currency_symbol),
                help=(
                    "Model estimate for the next available trading session, "
                    "not a guaranteed price."
                ),
            )

            f2.metric(
                "Expected difference",
                format_price(forecast_change, currency_symbol),
            )

            f3.metric(
                "Expected percentage difference",
                format_percent(forecast_pct),
            )

            st.caption(
                f"Estimated session: {forecast_date} · "
                f"Model data through "
                f"{overview_report.get('latest_market_data_date', 'Unknown')}. "
                "For validation metrics and candidate comparisons, open the "
                "Forecast & Validation tab."
            )

            # Show a separate ML estimate when one is available.
            ml_price = overview_report.get(
                "ml_predicted_next_closing_price"
            )

            ml_change = overview_report.get("ml_expected_price_change")
            ml_pct = overview_report.get("ml_expected_percentage_change")
            ml_name = overview_report.get("best_ml_model_name")

            if ml_price is not None and ml_name:
                st.markdown(
                    f"**Best ML candidate (separate comparison):** {ml_name}"
                )
                m1, m2, m3 = st.columns([1.5, 1, 1])

                m1.metric(
                    "ML estimated next close",
                    format_price(ml_price, currency_symbol),
                )

                m2.metric(
                    "ML difference vs latest close",
                    format_price(ml_change, currency_symbol),
                )

                m3.metric(
                    "ML percentage difference",
                    format_percent(ml_pct),
                )

                st.caption(
                    "This is the best machine-learning candidate by "
                    "cross-validation MAE. It is not guaranteed to be "
                    "more accurate than the baseline."
                )

        elif overview_error:
            st.warning(
                "Historical prices loaded, but the next-day forecast "
                "could not be generated: "
                f"{overview_error} Open Forecast & Validation for details."
            )

        else:
            st.info(
                "The next-day estimate is generated when you click "
                "**Analyze selected stock**. If it is unavailable, open "
                "Forecast & Validation to check the model status."
            )

        st.divider()

        chart_title, chart_count = st.columns([3, 1])

        with chart_title:
            st.markdown("#### Closing-price history")

        with chart_count:
            st.caption(f"{len(history):,} daily rows")

        # Prepare a normal datetime column for the chart. This avoids
        # plotting dates as hourly timestamps or as an implicit row index.
        chart_data = history[["Close"]].copy()

        chart_data = chart_data.rename(
            columns={"Close": "Closing Price"}
        )

        if getattr(chart_data.index, "tz", None) is not None:
            chart_data.index = chart_data.index.tz_localize(None)

        chart_data.index.name = "Date"
        chart_data = chart_data.sort_index().reset_index()

        chart_data["Date"] = pd.to_datetime(
            chart_data["Date"],
            errors="coerce",
        )

        chart_data = chart_data.dropna(
            subset=["Date", "Closing Price"]
        )

        # Display readable trading dates instead of hourly-looking ticks.
        chart_data["Date"] = chart_data["Date"].dt.strftime(
            "%d %b %Y"
        )

        st.line_chart(
            chart_data,
            x="Date",
            y="Closing Price",
            x_label="Trading date",
            y_label=f"Closing price ({currency_symbol})",
            height=380,
            width="stretch",
        )


        st.markdown("#### Latest trading session")

        o, h, l = st.columns(3)

        o.metric(
            "Open",
            format_price(latest_row["Open"], currency_symbol),
        )

        h.metric(
            "Day's high",
            format_price(latest_row["High"], currency_symbol),
        )

        l.metric(
            "Day's low",
            format_price(latest_row["Low"], currency_symbol),
        )

        st.caption(
            "These values belong to the latest available daily bar; "
            "the close is not necessarily a live intraday quote."
        )
            # --------------------------------------------------
    # 10. FORECAST AND VALIDATION TAB
    # --------------------------------------------------
    with forecast_tab:
        st.markdown("#### Next-session closing-price estimate")

        report = st.session_state.get("loaded_prediction_report")

        prediction_error = st.session_state.get(
            "loaded_prediction_error",
            "",
        )

        if report:
            if report.get("forecast_mode") == "baseline_only":
                st.warning(
                    "Insufficient history for ML training: only "
                    f"{report.get('available_ohlc_rows', 0)} valid daily "
                    "OHLC rows were returned. This is a clearly labelled "
                    "last-close baseline; no machine-learning forecast was "
                    "produced. Try a ticker with a longer trading history."
                )

            predicted = report.get("predicted_next_closing_price")
            model_close = report.get("latest_closing_price")
            expected_change = report.get("expected_price_change")
            expected_pct = report.get("expected_percentage_change")
            validation = report.get("validation") or {}

            model_col, date_col = st.columns([3, 1])

            with model_col:
                st.markdown(
                    f"**Selected approach:** "
                    f"{report.get('model_name', 'Unknown')}"
                )

                st.caption(
                    "Forecast reference close: "
                    f"{format_price(model_close, currency_symbol)}"
                )

            with date_col:
                st.caption("Estimated next session")
                st.markdown(
                    f"**{report.get('estimated_next_trading_date', 'Unknown')}**"
                )

            p1, p2, p3 = st.columns(3)

            p1.metric(
                "Estimated closing price",
                format_price(predicted, currency_symbol),
            )

            p2.metric(
                "Difference vs latest close",
                format_price(expected_change, currency_symbol),
            )

            p3.metric(
                "Estimated percentage difference",
                format_percent(expected_pct),
            )

            # Show the ML forecast separately for transparent comparison.
            ml_predicted = report.get("ml_predicted_next_closing_price")
            ml_change = report.get("ml_expected_price_change")
            ml_pct = report.get("ml_expected_percentage_change")
            ml_name = report.get("best_ml_model_name")

            if ml_predicted is not None and ml_name:
                st.divider()
                st.markdown(
                    f"#### Best machine-learning candidate: {ml_name}"
                )

                st.caption(
                    "This estimate is provided alongside the primary "
                    "forecast so you can compare a trained ML model with "
                    "the last-close baseline."
                )

                ml1, ml2, ml3 = st.columns(3)

                ml1.metric(
                    "ML estimated closing price",
                    format_price(ml_predicted, currency_symbol),
                )

                ml2.metric(
                    "ML difference vs latest close",
                    format_price(ml_change, currency_symbol),
                )

                ml3.metric(
                    "ML percentage difference",
                    format_percent(ml_pct),
                )

            st.caption(
                f"Model data through "
                f"{report.get('latest_market_data_date', 'Unknown')} · "
                f"{report.get('training_rows', 'N/A')} supervised rows · "
                f"{report.get('feature_count', 'N/A')} features · "
                "Model trained on up to five years of daily data"
            )

            st.divider()
            st.markdown("#### Chronological holdout validation")

            # Modified metrics: show ML, baseline, and selected approach
            # separately so the primary forecast is not confused with
            # the best machine-learning candidate.
            v1, v2, v3 = st.columns(3)

            v1.metric(
                "Best ML Model MAE",
                format_price(
                    validation.get("best_ml_model_mae"),
                    currency_symbol,
                ),
                help=(
                    "Mean absolute error of the best ML candidate "
                    "on the chronological holdout. Lower is better."
                ),
            )

            v2.metric(
                "Last-Close Baseline MAE",
                format_price(
                    validation.get("last_close_baseline_mae"),
                    currency_symbol,
                ),
                help=(
                    "Mean absolute error when predicting that the next "
                    "close equals the latest observed close."
                ),
            )

            v3.metric(
                "Selected Approach MAE",
                format_price(
                    validation.get("selected_model_mae"),
                    currency_symbol,
                ),
                help=(
                    "Mean absolute error of the approach selected "
                    "for the primary forecast."
                ),
            )

            if validation.get("selected_approach_is_baseline") is True:
                st.info(
                    "The Last-Close Baseline was selected because the ML "
                    "candidate did not outperform it in the current "
                    "holdout comparison. The displayed estimate therefore "
                    "equals the latest observed close."
                )
            elif report.get("ml_forecast_is_primary") is True:
                st.success(
                    "The machine-learning forecast is the primary estimate "
                    "because it outperformed the Last-Close Baseline in the "
                    "current chronological holdout comparison."
                )
            else:
                st.info(
                    "The report does not include enough validation details "
                    "to describe the baseline comparison."
                )

            with st.expander(
                "Compare candidate approaches (cross-validation)"
            ):
                candidate_scores = validation.get(
                    "candidate_scores",
                    [],
                )

                if candidate_scores:
                    st.caption(
                        "These are chronological development-fold scores, "
                        "not final holdout scores. Lower MAE/RMSE is better; "
                        "higher R² is generally better."
                    )

                    st.dataframe(
                        make_candidate_comparison(candidate_scores),
                        hide_index=True,
                        width="stretch",
                    )

                else:
                    st.info(
                        "Candidate-model cross-validation scores "
                        "are unavailable."
                    )

            with st.expander(
                "Validation methodology and additional metrics"
            ):
                st.write(
                    "Machine-learning candidates are compared using "
                    "chronological time-series cross-validation. The selected "
                    "ML candidate is compared with the Last-Close Baseline "
                    "on a later chronological holdout to determine the "
                    "displayed primary approach. Because this comparison "
                    "influences the displayed approach, the holdout is a "
                    "diagnostic comparison rather than a completely "
                    "untouched final evaluation. The selected ML candidate "
                    "is refitted on all available supervised rows for the "
                    "latest forecast."
                )

                left, right = st.columns(2)

                with left:
                    st.write(
                        f"Development rows: "
                        f"{validation.get('train_rows', 'N/A')}"
                    )

                    st.write(
                        f"Holdout rows: "
                        f"{validation.get('test_rows', 'N/A')}"
                    )

                    st.write(
                        "Cross-validation MAE: "
                        f"{format_price(validation.get('cross_validation_mae'), currency_symbol)}"
                    )

                    st.write(
                        "Cross-validation RMSE: "
                        f"{format_price(validation.get('cross_validation_rmse'), currency_symbol)}"
                    )

                with right:
                    st.write(
                        "Cross-validation R²: "
                        f"{validation.get('cross_validation_r2', 'N/A')}"
                    )

                    st.write(
                        "Baseline holdout RMSE: "
                        f"{format_price(validation.get('last_close_baseline_rmse'), currency_symbol)}"
                    )

                    st.write(
                        "Selected approach holdout R²: "
                        f"{validation.get('selected_model_r2', 'N/A')}"
                    )

                    improvement = validation.get(
                        "mae_improvement_over_baseline_pct"
                    )

                    st.write(
                        "MAE improvement vs baseline: "
                        f"{format_percent(improvement) if improvement is not None else 'N/A'}"
                    )

            with st.expander("How to interpret the metrics"):
                st.markdown(
                    """
                    - **MAE:** average absolute difference between predicted
                      and actual closing prices.
                    - **RMSE:** gives more weight to larger errors than MAE.
                    - **R²:** compares predictions with a mean-based reference
                      and can be negative on unseen data.
                    - **Cross-validation:** compares ML candidates on
                      chronological development folds.
                    - **Holdout comparison:** compares the chosen ML candidate
                      with the Last-Close Baseline on a later chronological
                      period. Because this comparison influences which
                      approach is displayed as primary, it is a diagnostic
                      comparison rather than a completely untouched final test.
                    - **Last-Close Baseline:** predicts that the next close
                      will equal the latest observed close.
                    """
                )

            st.warning(
                "Educational forecast only, not financial advice. The "
                "estimated date uses business days and does not account "
                "for exchange holidays. Historical validation cannot "
                "guarantee future performance."
            )

        elif prediction_error:
            st.warning(
                "Historical prices loaded, but the next-session model "
                "could not be trained: "
                f"{prediction_error}"
            )

        else:
            st.info("No forecast report is available for this selection.")

    # --------------------------------------------------
    # 11. HISTORICAL DATA TAB
    # --------------------------------------------------
    with data_tab:
        st.markdown("#### Inspect or download historical data")

        st.caption(
            f"{loaded_ticker} · {len(history):,} daily rows · "
            f"Selected period: {loaded_period}"
        )

        display_history = history.copy()

        if getattr(display_history.index, "tz", None) is not None:
            display_history.index = display_history.index.tz_localize(None)

        display_history.index.name = "Date"

        st.dataframe(
            display_history.sort_index(ascending=False),
            width="stretch",
        )

        st.download_button(
            "Download historical data (CSV)",
            data=display_history.to_csv().encode("utf-8"),
            file_name=f"{sanitize_filename(loaded_ticker)}_history.csv",
            mime="text/csv",
        )

else:
    # --------------------------------------------------
    # 12. EMPTY / WELCOME STATE
    # --------------------------------------------------
    st.markdown(
        """
        <div class="terminal-panel">
          <div class="terminal-eyebrow">READY FOR ANALYSIS</div>
          <h3 style="margin-top:8px">
            Your market workspace is ready.
          </h3>
          <p style="color:#91a0af;line-height:1.65">
            Choose a market in the sidebar, search for a company or ticker,
            select the exact exchange listing, and run the analysis. The
            dashboard will show historical prices, a next-session estimate,
            and time-ordered validation metrics.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    a, b, c = st.columns(3)

    with a:
        st.markdown("#### 01 / Find")
        st.caption(
            "Search supported Indian or US listings by name or ticker."
        )

    with b:
        st.markdown("#### 02 / Analyze")
        st.caption(
            "Load daily OHLC history and build a ticker-specific forecast."
        )

    with c:
        st.markdown("#### 03 / Evaluate")
        st.caption(
            "Review holdout errors, candidate comparison, and the baseline."
        )

    st.info(
        "Start with the controls in the left sidebar. Yahoo Finance data "
        "timing and exchange coverage may vary."
    )


# --------------------------------------------------
# 13. METHODOLOGY AND FOOTER
# --------------------------------------------------
with st.expander("About the model, data, and limitations"):
    st.markdown(
        """
        **Model selection and evaluation**

        Ridge Regression, Random Forest, and Histogram Gradient
        Boosting are compared using chronological time-series
        cross-validation. The ML candidate with the lowest average
        cross-validation MAE is compared with the Last-Close Baseline
        on a later chronological holdout. The displayed primary
        approach is selected using that comparison. Since the holdout
        comparison influences the displayed approach, it should be
        interpreted as a diagnostic comparison rather than a completely
        untouched final evaluation. When an ML forecast is available,
        the best ML candidate is also displayed separately for comparison.

        The selected ML candidate is refitted on all available supervised
        rows to generate the latest estimate. The Last-Close Baseline
        predicts that the next closing price will equal the latest
        observed closing price.

        **Limitations**

        Historical prices can be delayed, missing, or affected by corporate
        actions. The estimated next-session date uses business days and does
        not account for exchange holidays. Historical model errors do not
        guarantee future performance. This is an educational project, not
        financial advice.
        """
    )

st.divider()

st.caption(
    "STOCK PRICE INTELLIGENCE SYSTEM · EDUCATIONAL PROJECT · "
    "Yahoo Finance data timing and exchange coverage may vary. "
    "Historical performance does not guarantee future returns. "
    "Not financial advice."
)