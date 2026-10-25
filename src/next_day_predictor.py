
"""Train and evaluate next-session closing-price forecasts.

Uses information available at each session close to predict the next
available session's close.

Candidate approaches:
- Ridge Regression
- Random Forest
- Histogram Gradient Boosting
- Last-Close Baseline

The best ML candidate is selected using chronological cross-validation MAE.
Its chronological holdout performance is compared with the last-close
baseline to choose the primary forecast.

Note:
The holdout-based selection rule is a practical diagnostic, not an
unbiased final estimate of the complete model-selection procedure.

Volume is excluded so this pipeline also supports indices without volume.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MINIMUM_RAW_ROWS = 150
MINIMUM_SUPERVISED_ROWS = 100
MINIMUM_DEVELOPMENT_ROWS = 60
MINIMUM_TEST_ROWS = 20

CV_SPLITS = 3
CV_GAP = 1

BASELINE_NAME = "Last-Close Baseline"


def _normalize_columns(history: pd.DataFrame) -> pd.DataFrame:
    """Normalize Yahoo Finance columns, including MultiIndex columns."""
    df = history.copy()

    if isinstance(df.columns, pd.MultiIndex):
        standard = {
            "Open", "High", "Low", "Close",
            "Adj Close", "Volume", "Dividends", "Stock Splits",
        }
        flattened = []

        for column in df.columns:
            parts = [str(part) for part in column]
            matches = [part for part in parts if part in standard]
            flattened.append(
                matches[0] if matches else "_".join(parts)
            )

        df.columns = flattened

    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]
    return df


def _clean_ohlcv(history: pd.DataFrame) -> pd.DataFrame:
    """Validate daily OHLC data. Volume is optional."""
    if history is None or history.empty:
        raise ValueError("No historical data was supplied.")

    df = _normalize_columns(history)
    required = ["Open", "High", "Low", "Close"]
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            "Historical data is missing required columns: "
            + ", ".join(sorted(missing))
        )

    raw = df[required].copy()
    raw = raw.apply(pd.to_numeric, errors="coerce")
    raw = raw.replace([np.inf, -np.inf], np.nan)
    raw = raw.dropna(subset=required)
    raw = raw.loc[(raw[required] > 0).all(axis=1)].copy()
    raw = raw.sort_index()
    raw = raw.loc[~raw.index.duplicated(keep="last")]

    if raw.empty:
        raise ValueError("No valid OHLC rows remain after cleaning.")

    if not raw.index.is_monotonic_increasing:
        raise ValueError("Historical data must be sorted chronologically.")

    return raw


def _make_features(history: pd.DataFrame) -> pd.DataFrame:
    """Build features using only information available at each session."""
    raw = _clean_ohlcv(history)
    close = raw["Close"]

    features = pd.DataFrame(index=raw.index)

    features["Open"] = raw["Open"]
    features["High"] = raw["High"]
    features["Low"] = raw["Low"]
    features["Close"] = close

    for lag in [1, 2, 3, 5, 10]:
        features[f"Close_Lag_{lag}"] = close.shift(lag)

    features["Return_1D"] = close.pct_change(fill_method=None)
    features["Return_5D"] = close.pct_change(5, fill_method=None)

    for window in [5, 10, 20]:
        features[f"MA_{window}"] = close.rolling(
            window=window,
            min_periods=window,
        ).mean()

    for span in [10, 20]:
        features[f"EMA_{span}"] = close.ewm(
            span=span,
            adjust=False,
        ).mean()

    features["Volatility_5D"] = features["Return_1D"].rolling(
        5,
        min_periods=5,
    ).std()

    features["Daily_Range_Pct"] = (
        (raw["High"] - raw["Low"]) / close
    )

    previous_close = close.shift(1)
    features["Open_Gap_Pct"] = (
        (raw["Open"] - previous_close) / previous_close
    )

    return features.replace([np.inf, -np.inf], np.nan)


def _build_candidates() -> dict:
    """Create fresh machine-learning candidate estimators."""
    return {
        "Ridge Regression": make_pipeline(
            StandardScaler(),
            Ridge(alpha=10.0),
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),

        "Histogram Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
    }


def _calculate_metrics(actual, predicted) -> dict:
    """Calculate MAE, RMSE, and R² after validating arrays."""
    actual_values = np.asarray(actual, dtype=float).reshape(-1)
    predicted_values = np.asarray(predicted, dtype=float).reshape(-1)

    if len(actual_values) != len(predicted_values):
        raise ValueError(
            "Actual and predicted arrays have different lengths."
        )

    if len(actual_values) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    if not (
        np.isfinite(actual_values).all()
        and np.isfinite(predicted_values).all()
    ):
        raise ValueError("Metrics contain non-finite values.")

    return {
        "mae": float(
            mean_absolute_error(actual_values, predicted_values)
        ),
        "rmse": float(
            np.sqrt(mean_squared_error(actual_values, predicted_values))
        ),
        "r2": (
            float(r2_score(actual_values, predicted_values))
            if len(actual_values) >= 2
            else None
        ),
    }


def _cross_validate_candidates(
    X_development: pd.DataFrame,
    y_development: pd.Series,
) -> list[dict]:
    """Compare ML candidates and baseline on chronological folds."""
    splitter = TimeSeriesSplit(
        n_splits=CV_SPLITS,
        gap=CV_GAP,
    )

    scores = []
    baseline_maes = []
    baseline_rmses = []
    baseline_r2s = []

    # Evaluate the baseline on the same validation folds.
    for _, validation_indices in splitter.split(X_development):
        X_validation = X_development.iloc[validation_indices]
        y_validation = y_development.iloc[validation_indices]

        metrics = _calculate_metrics(
            y_validation,
            X_validation["Close"].to_numpy(),
        )

        baseline_maes.append(metrics["mae"])
        baseline_rmses.append(metrics["rmse"])

        if metrics["r2"] is not None:
            baseline_r2s.append(metrics["r2"])

    scores.append({
        "model": BASELINE_NAME,
        "mae": float(np.mean(baseline_maes)),
        "rmse": float(np.mean(baseline_rmses)),
        "r2": (
            float(np.mean(baseline_r2s))
            if baseline_r2s
            else None
        ),
        "fold_maes": baseline_maes,
        "fold_rmses": baseline_rmses,
    })

    # Evaluate ML candidates using chronological folds.
    for name, prototype in _build_candidates().items():
        fold_maes = []
        fold_rmses = []
        fold_r2s = []

        try:
            for train_indices, validation_indices in splitter.split(
                X_development
            ):
                X_train = X_development.iloc[train_indices]
                y_train = y_development.iloc[train_indices]

                X_validation = X_development.iloc[validation_indices]
                y_validation = y_development.iloc[validation_indices]

                model = clone(prototype)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_validation)

                metrics = _calculate_metrics(
                    y_validation,
                    predictions,
                )

                fold_maes.append(metrics["mae"])
                fold_rmses.append(metrics["rmse"])

                if metrics["r2"] is not None:
                    fold_r2s.append(metrics["r2"])

            if fold_maes:
                scores.append({
                    "model": name,
                    "mae": float(np.mean(fold_maes)),
                    "rmse": float(np.mean(fold_rmses)),
                    "r2": (
                        float(np.mean(fold_r2s))
                        if fold_r2s
                        else None
                    ),
                    "fold_maes": fold_maes,
                    "fold_rmses": fold_rmses,
                })

        except Exception as exc:
            warnings.warn(
                f"{name} failed during cross-validation: {exc}",
                RuntimeWarning,
            )

    ml_scores = [
        score for score in scores
        if score["model"] != BASELINE_NAME
    ]

    if not ml_scores:
        raise RuntimeError(
            "All machine-learning candidates failed during cross-validation."
        )

    return sorted(scores, key=lambda item: item["mae"])


def _price_change(
    predicted: float,
    latest: float,
) -> tuple[float, float]:
    """Return absolute and percentage change from the latest close."""
    change = float(predicted - latest)
    percentage = float(change / latest * 100.0)
    return change, percentage


def train_and_predict_next_close(history: pd.DataFrame) -> dict:
    """Train candidates and return the selected next-close forecast."""
    raw = _clean_ohlcv(history)

    if len(raw) < MINIMUM_RAW_ROWS:
        raise ValueError(
            f"Only {len(raw)} valid OHLC rows are available; "
            f"at least {MINIMUM_RAW_ROWS} are required. "
            "Download a longer historical period."
        )

    features = _make_features(raw)

    # Features through session t predict the close at session t+1.
    target = raw["Close"].shift(-1).rename("Target_Next_Close")

    supervised = (
        features.join(target)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(supervised) < MINIMUM_SUPERVISED_ROWS:
        raise ValueError(
            f"Only {len(supervised)} complete supervised rows remain. "
            "A longer or cleaner history is required."
        )

    feature_columns = list(features.columns)
    X = supervised[feature_columns]
    y = supervised["Target_Next_Close"]

    # Preserve chronological order: the last 20% is held out.
    test_start = int(len(supervised) * 0.8)
    development_end = test_start - CV_GAP

    X_development = X.iloc[:development_end]
    y_development = y.iloc[:development_end]
    X_test = X.iloc[test_start:]
    y_test = y.iloc[test_start:]

    if (
        len(X_development) < MINIMUM_DEVELOPMENT_ROWS
        or len(X_test) < MINIMUM_TEST_ROWS
    ):
        raise ValueError(
            "Not enough rows for the development and holdout periods."
        )

    # Select the best ML candidate by development-period CV MAE.
    candidate_scores = _cross_validate_candidates(
        X_development,
        y_development,
    )

    ml_scores = [
        score for score in candidate_scores
        if score["model"] != BASELINE_NAME
    ]

    if not ml_scores:
        raise RuntimeError(
            "All machine-learning candidates failed during cross-validation."
        )

    ml_scores = sorted(ml_scores, key=lambda item: item["mae"])
    best_ml_score = ml_scores[0]
    best_ml_name = best_ml_score["model"]
    prototypes = _build_candidates()

    # Baseline holdout: predict the next close using the current close.
    baseline_predictions = X_test["Close"].to_numpy()
    baseline_metrics = _calculate_metrics(
        y_test,
        baseline_predictions,
    )
    baseline_mae = baseline_metrics["mae"]

    # Evaluate the selected ML candidate on the same holdout.
    test_model = clone(prototypes[best_ml_name])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_model.fit(X_development, y_development)
        test_predictions = test_model.predict(X_test)

    test_metrics = _calculate_metrics(y_test, test_predictions)

    improvement_pct = (
        (baseline_mae - test_metrics["mae"]) / baseline_mae * 100.0
        if baseline_mae > 0
        else None
    )

    # This comparison is diagnostic; it is not an unbiased final
    # estimate after using holdout performance to choose the approach.
    ml_beats_baseline = bool(test_metrics["mae"] < baseline_mae)
    selected_approach_is_baseline = not ml_beats_baseline

    # Refit the best ML candidate on all available supervised rows.
    final_model = clone(prototypes[best_ml_name])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model.fit(X, y)

    latest_features = features.iloc[[-1]][feature_columns]

    if latest_features.isna().any().any():
        missing_features = latest_features.columns[
            latest_features.isna().any()
        ].tolist()

        raise ValueError(
            "Latest feature values are incomplete: "
            + ", ".join(missing_features)
        )

    latest_date = pd.Timestamp(latest_features.index[-1])
    latest_close = float(latest_features["Close"].iloc[0])

    if not np.isfinite(latest_close) or latest_close <= 0:
        raise ValueError(
            "Latest closing price must be positive and finite."
        )

    # Preserve the ML prediction even when the baseline is primary.
    predicted_close = float(final_model.predict(latest_features)[0])

    if not np.isfinite(predicted_close) or predicted_close <= 0:
        raise ValueError(
            "The selected ML model produced an invalid price estimate."
        )

    price_change, percentage_change = _price_change(
        predicted_close,
        latest_close,
    )

    # Weekday approximation; exchange holidays are not considered.
    next_date = latest_date + pd.offsets.BDay(1)

    # Choose the primary forecast using the holdout comparison.
    if selected_approach_is_baseline:
        primary_model_name = BASELINE_NAME
        primary_predicted_close = latest_close
        primary_price_change = 0.0
        primary_percentage_change = 0.0
        primary_metrics = baseline_metrics
    else:
        primary_model_name = best_ml_name
        primary_predicted_close = predicted_close
        primary_price_change = price_change
        primary_percentage_change = percentage_change
        primary_metrics = test_metrics

    return {
        # Primary forecast for the dashboard.
        "model_name": primary_model_name,
        "predicted_next_closing_price": primary_predicted_close,
        "latest_closing_price": latest_close,
        "expected_price_change": primary_price_change,
        "expected_percentage_change": primary_percentage_change,
        "latest_market_data_date": latest_date.strftime("%d %B %Y"),
        "estimated_next_trading_date": next_date.strftime("%d %B %Y"),
        "training_rows": int(len(supervised)),
        "feature_count": int(len(feature_columns)),

        # Keep the ML prediction separately available for comparison.
        "best_ml_model_name": best_ml_name,
        "ml_predicted_next_closing_price": predicted_close,
        "ml_expected_price_change": price_change,
        "ml_expected_percentage_change": percentage_change,
        "ml_forecast_is_primary": bool(not selected_approach_is_baseline),

        "validation": {
            "train_rows": int(len(X_development)),
            "test_rows": int(len(X_test)),

            # Metrics for the selected primary approach.
            "selected_model_mae": primary_metrics["mae"],
            "selected_model_rmse": primary_metrics["rmse"],
            "selected_model_r2": primary_metrics["r2"],

            # Best ML candidate's development-period CV metrics.
            "cross_validation_mae": best_ml_score["mae"],
            "cross_validation_rmse": best_ml_score["rmse"],
            "cross_validation_r2": best_ml_score["r2"],

            # Baseline holdout metrics.
            "last_close_baseline_mae": baseline_mae,
            "last_close_baseline_rmse": baseline_metrics["rmse"],
            "last_close_baseline_r2": baseline_metrics["r2"],

            "mae_improvement_over_baseline_pct": improvement_pct,
            "beats_last_close_baseline": ml_beats_baseline,
            "selected_approach_is_baseline": bool(
                selected_approach_is_baseline
            ),

            "candidate_scores": candidate_scores,

            # ML-only holdout metrics remain separately accessible.
            "best_ml_model_mae": test_metrics["mae"],
            "best_ml_model_rmse": test_metrics["rmse"],
            "best_ml_model_r2": test_metrics["r2"],
            "best_ml_cross_validation_mae": best_ml_score["mae"],
            "best_ml_cross_validation_rmse": best_ml_score["rmse"],
            "best_ml_cross_validation_r2": best_ml_score["r2"],
            "best_ml_mae_improvement_over_baseline_pct": improvement_pct,
            "best_ml_beats_last_close_baseline": ml_beats_baseline,

            "evaluation_method": (
                "Chronological holdout with one-step-ahead predictions "
                "using features available at each session close"
            ),
            "cv_splits": CV_SPLITS,
            "cv_gap": CV_GAP,
        },
    }
