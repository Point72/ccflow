#!/usr/bin/env python
"""
ML Pipeline Demo: Smart Model Selection
========================================

This is the example from the original design conversation — a realistic ML
pipeline that demonstrates how Flow.model lets you write plain functions,
wire them by passing outputs as inputs, and execute with .flow.compute().

Features demonstrated:
    1. @Flow.model with auto-wrap (plain return types, no GenericResult needed)
    2. Lazy[T] for conditional evaluation (skip slow model if fast is good enough)
    3. .flow.compute() for execution with automatic context propagation
    4. .flow.with_inputs() for context transforms (lookback windows)
    5. Factored wiring — build_pipeline() shows how to reuse the same graph
       structure with different data sources

The pipeline:

    load_dataset ──> prepare_features ──> train_linear ──> evaluate ──> fast_metrics ──┐
                                     └──> train_forest ──> evaluate ──> slow_metrics ──┴──> smart_training

Run with: python examples/ml_pipeline_demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from math import sin

from ccflow import Flow, Lazy


# =============================================================================
# Domain types (stand-ins for real ML objects)
# =============================================================================


@dataclass
class PreparedData:
    """Container for train/test split data."""

    X_train: list  # list of feature vectors
    X_test: list
    y_train: list  # list of target values
    y_test: list


@dataclass
class TrainedModel:
    """A fitted model (placeholder)."""

    name: str
    coefficients: list
    intercept: float
    augment: bool  # Whether to add sin feature during prediction


@dataclass
class Metrics:
    """Evaluation metrics."""

    r2: float
    mse: float
    model_name: str


# =============================================================================
# Data Loading
# =============================================================================


@Flow.model
def load_dataset(start_date: date, end_date: date, source: str = "warehouse") -> list:
    """Load raw dataset for a date range.

    Returns a list of dicts (standing in for a DataFrame).
    Auto-wrapped: returns plain list, framework wraps in GenericResult.
    """
    n_days = (end_date - start_date).days + 1
    print(f"  [load_dataset] Loading {n_days} days from '{source}' ({start_date} to {end_date})")
    # True relationship: target = 2.0 * x + 10.0 + 15.0 * sin(x * 0.2)
    # Linear model captures the trend (R^2 ~0.93), forest also captures the sin wave (~0.99)
    return [
        {
            "date": str(start_date + timedelta(days=i)),
            "x": float(i),
            "target": 2.0 * i + 10.0 + 15.0 * sin(i * 0.2),
        }
        for i in range(n_days)
    ]


# =============================================================================
# Feature Engineering
# =============================================================================


@Flow.model
def prepare_features(raw_data: list) -> PreparedData:
    """Split data into train/test.

    Returns a PreparedData dataclass — the framework auto-wraps it in GenericResult.
    Downstream models can request individual fields via prepared["X_train"] etc.
    """
    n = len(raw_data)
    split = int(n * 0.8)
    print(f"  [prepare_features] {n} rows, split at {split}")

    X = [[r["x"]] for r in raw_data]
    y = [r["target"] for r in raw_data]

    return PreparedData(
        X_train=X[:split],
        X_test=X[split:],
        y_train=y[:split],
        y_test=y[split:],
    )


# =============================================================================
# Model Training
# =============================================================================


def _ols_fit(X, y):
    """Simple OLS: compute coefficients and intercept."""
    n = len(X)
    n_feat = len(X[0])
    y_mean = sum(y) / n
    x_means = [sum(row[j] for row in X) / n for j in range(n_feat)]

    coefficients = []
    for j in range(n_feat):
        cov = sum((X[i][j] - x_means[j]) * (y[i] - y_mean) for i in range(n)) / n
        var = sum((X[i][j] - x_means[j]) ** 2 for i in range(n)) / n
        coefficients.append(cov / var if var > 1e-10 else 0.0)

    intercept = y_mean - sum(c * m for c, m in zip(coefficients, x_means))
    return coefficients, intercept


def _augment(X):
    """Add sin(x*0.2) feature to capture non-linearity."""
    return [row + [sin(row[0] * 0.2)] for row in X]


@Flow.model
def train_linear(prepared: PreparedData) -> TrainedModel:
    """Train a fast linear model (linear features only)."""
    print(f"  [train_linear] Fitting on {len(prepared.X_train)} samples")
    coefficients, intercept = _ols_fit(prepared.X_train, prepared.y_train)
    return TrainedModel(name="LinearRegression", coefficients=coefficients, intercept=intercept, augment=False)


@Flow.model
def train_forest(prepared: PreparedData, n_estimators: int = 100) -> TrainedModel:
    """Train a model that also captures non-linear patterns (simulated)."""
    print(f"  [train_forest] Fitting {n_estimators} trees on {len(prepared.X_train)} samples")
    # Augment with sin feature to capture non-linearity
    X_aug = _augment(prepared.X_train)
    coefficients, intercept = _ols_fit(X_aug, prepared.y_train)
    return TrainedModel(
        name=f"RandomForest(n={n_estimators})",
        coefficients=coefficients,
        intercept=intercept,
        augment=True,
    )


# =============================================================================
# Model Evaluation
# =============================================================================


@Flow.model
def evaluate_model(model: TrainedModel, prepared: PreparedData) -> Metrics:
    """Evaluate a trained model on test data."""
    X_test = prepared.X_test
    y_test = prepared.y_test
    X_eval = _augment(X_test) if model.augment else X_test

    y_pred = [
        model.intercept + sum(c * x for c, x in zip(model.coefficients, row))
        for row in X_eval
    ]

    y_mean = sum(y_test) / len(y_test) if y_test else 0
    ss_tot = sum((y - y_mean) ** 2 for y in y_test) or 1
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_test, y_pred))
    r2 = 1.0 - ss_res / ss_tot
    mse = ss_res / len(y_test) if y_test else 0

    print(f"  [evaluate_model] {model.name}: R^2={r2:.4f}, MSE={mse:.2f}")
    return Metrics(r2=r2, mse=mse, model_name=model.name)


# =============================================================================
# Smart Pipeline with Conditional Execution
# =============================================================================


@Flow.model
def smart_training(
    # data: PreparedData,
    fast_metrics: Metrics,
    slow_metrics: Lazy[Metrics],  # Only evaluated if fast isn't good enough
    threshold: float = 0.9,
) -> Metrics:
    """Use fast model if good enough, else fall back to slow.

    The slow_metrics parameter is Lazy — it receives a zero-arg thunk.
    If the fast model exceeds the threshold, the slow model is never
    trained or evaluated at all.
    """
    print(f"  [smart_training] Fast R^2={fast_metrics.r2:.4f}, threshold={threshold}")
    if fast_metrics.r2 >= threshold:
        print("  [smart_training] Fast model is good enough! Skipping slow model.")
        return fast_metrics
    else:
        print("  [smart_training] Fast model below threshold, evaluating slow model...")
        return slow_metrics()


# =============================================================================
# Pipeline Wiring Helper
# =============================================================================


def build_pipeline(raw, *, n_estimators=200, threshold=0.95):
    """Wire a complete train/evaluate/select pipeline from a data source.

    This function shows the flexibility of the approach: the same wiring
    logic can be applied to different data sources (raw, lookback_raw, etc.)
    without duplicating code. Everything here is just wiring — no computation
    happens until .flow.compute() is called.

    Args:
        raw: A CallableModel or BoundModel that produces raw data (list of dicts)
        n_estimators: Number of trees for the forest model
        threshold: R^2 threshold for the fast/slow model selection

    Returns:
        A smart_training model instance ready for .flow.compute()
    """
    # Feature engineering — returns a PreparedData with X_train, X_test, etc.
    prepared = prepare_features(raw_data=raw)

    # Train both models — each receives the whole PreparedData and extracts
    # the fields it needs internally.
    linear = train_linear(prepared=prepared)
    forest = train_forest(prepared=prepared, n_estimators=n_estimators)

    # Evaluate both
    linear_metrics = evaluate_model(model=linear, prepared=prepared)
    forest_metrics = evaluate_model(model=forest, prepared=prepared)

    # Smart selection with Lazy — forest is only evaluated if linear isn't good enough
    return smart_training(
        fast_metrics=linear_metrics,
        slow_metrics=forest_metrics,
        threshold=threshold,
    )


# =============================================================================
# Main: Wire and execute the pipeline
# =============================================================================


def main():
    print("=" * 70)
    print("ML Pipeline Demo: Smart Model Selection with Flow.model")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Wire the pipeline (no computation happens here)
    # ------------------------------------------------------------------
    print("\n--- Wiring the pipeline (lazy, no computation yet) ---\n")

    raw = load_dataset(source="prod_warehouse")

    # build_pipeline factors out the repeated wiring logic.
    # Linear R^2 ≈ 0.93. Threshold is 0.95 → falls through to forest.
    pipeline = build_pipeline(raw, n_estimators=200, threshold=0.95)

    print("Pipeline wired. No functions have been called yet.")

    # ------------------------------------------------------------------
    # Step 2: Execute — linear not good enough, falls back to forest
    # ------------------------------------------------------------------
    print("\n--- Executing pipeline (Jan-Jun 2024) ---\n")
    result = pipeline.flow.compute(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 30),
    )

    print(f"\n  Best model: {result.value.model_name}")
    print(f"  R^2: {result.value.r2:.4f}")
    print(f"  MSE: {result.value.mse:.2f}")

    # ------------------------------------------------------------------
    # Step 3: Context transforms (lookback) — reuse build_pipeline
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("With Lookback: Same pipeline structure, extra history for loading")
    print("=" * 70)

    # flow.with_inputs() creates a BoundModel that transforms the context
    # before calling the underlying model. start_date is shifted 30 days earlier.
    lookback_raw = raw.flow.with_inputs(
        start_date=lambda ctx: ctx.start_date - timedelta(days=30)
    )

    # Same wiring logic, different data source — no duplication.
    lookback_pipeline = build_pipeline(lookback_raw, n_estimators=200, threshold=0.95)

    print("\n--- Executing lookback pipeline ---\n")
    result2 = lookback_pipeline.flow.compute(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 30),
    )
    # Notice: load_dataset gets start_date=2023-12-02 (30 days earlier)

    print(f"\n  Best model: {result2.value.model_name}")
    print(f"  R^2: {result2.value.r2:.4f}")
    print(f"  MSE: {result2.value.mse:.2f}")

    # ------------------------------------------------------------------
    # Step 4: Lower threshold — linear is good enough, skip forest
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Lazy Evaluation: Lower threshold so fast model is good enough")
    print("=" * 70)

    # With threshold=0.80, the linear model's R^2 (~0.93) passes.
    # The forest is NEVER trained or evaluated — Lazy skips it entirely.
    fast_pipeline = build_pipeline(raw, n_estimators=200, threshold=0.80)

    print("\n--- Executing (slow model should NOT be trained) ---\n")
    result3 = fast_pipeline.flow.compute(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 30),
    )
    print(f"\n  Selected: {result3.value.model_name} (R^2={result3.value.r2:.4f})")
    print("  (Notice: train_forest and its evaluate_model were never called)")


if __name__ == "__main__":
    main()
