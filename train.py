"""
MLflow experiment tracking lab: California Housing price regression.

Changes from the original Lab1 (ElasticNet on Wine Quality):
  * Dataset: California Housing (20,640 rows, 8 features) instead of Wine Quality
  * Five model families compared, each with a small hyperparameter sweep
  * A parent run holds one nested child run per model configuration
  * 5-fold CV RMSE and training time logged alongside test metrics
  * Plots (predicted vs actual, residuals, feature importance) logged as artifacts
  * SQLite backend, so the Model Registry works locally
  * The best run is registered and given the "champion" alias
"""

import argparse
import os
import time
import warnings

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "california-housing-regression"
REGISTERED_MODEL_NAME = "CaliforniaHousingRegressor"
SEED = 42
TRUSTED_TREE_TYPES = [
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]

# (model name, constructor, list of hyperparameter configs)
SEARCH_SPACE = [
    ("ridge", lambda **p: make_pipeline(StandardScaler(), Ridge(**p)),
     [{"alpha": a} for a in (0.1, 1.0, 10.0)]),
    ("lasso", lambda **p: make_pipeline(StandardScaler(), Lasso(**p)),
     [{"alpha": a} for a in (0.001, 0.01, 0.1)]),
    ("random_forest", lambda **p: RandomForestRegressor(random_state=SEED, n_jobs=-1, **p),
     [{"n_estimators": 100, "max_depth": d} for d in (8, 16)]),
    ("gradient_boosting", lambda **p: GradientBoostingRegressor(random_state=SEED, **p),
     [{"n_estimators": 200, "learning_rate": lr, "max_depth": 3} for lr in (0.05, 0.1)]),
    ("hist_gradient_boosting", lambda **p: HistGradientBoostingRegressor(random_state=SEED, **p),
     [{"max_iter": 300, "learning_rate": lr} for lr in (0.05, 0.1)]),
]


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def plot_predictions(y_true, y_pred, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(y_true, y_pred, s=4, alpha=0.3)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", lw=1)
    axes[0].set(xlabel="Actual ($100k)", ylabel="Predicted ($100k)", title="Predicted vs actual")
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=60)
    axes[1].set(xlabel="Residual", ylabel="Count", title="Residual distribution")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def feature_importance(model, feature_names):
    """Return importances for tree models or |coefficients| for linear pipelines."""
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "feature_importances_"):
        values = est.feature_importances_
    elif hasattr(est, "coef_"):
        values = np.abs(est.coef_)
    else:
        return None
    return pd.Series(values, index=feature_names).sort_values()


def run_experiment(sample_frac):
    warnings.filterwarnings("ignore")
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = fetch_california_housing(as_frame=True)
    df = data.frame
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=SEED)
    X, y = df.drop(columns=["MedHouseVal"]), df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    results = []
    with mlflow.start_run(run_name="model-comparison") as parent:
        mlflow.set_tags({"dataset": "california_housing", "stage": "sweep"})
        mlflow.log_params({"n_train": len(X_train), "n_test": len(X_test), "sample_frac": sample_frac})

        for model_name, build, configs in SEARCH_SPACE:
            for params in configs:
                run_name = f"{model_name}-" + "-".join(f"{k}{v}" for k, v in params.items())
                with mlflow.start_run(run_name=run_name, nested=True) as child:
                    model = build(**params)

                    start = time.perf_counter()
                    model.fit(X_train, y_train)
                    train_time = time.perf_counter() - start

                    preds = model.predict(X_test)
                    rmse, mae, r2 = eval_metrics(y_test, preds)
                    cv_rmse = -cross_val_score(
                        build(**params), X_train, y_train, cv=5,
                        scoring="neg_root_mean_squared_error", n_jobs=-1,
                    ).mean()

                    mlflow.set_tag("model_family", model_name)
                    mlflow.log_params(params)
                    mlflow.log_metrics({
                        "rmse": rmse, "mae": mae, "r2": r2,
                        "cv_rmse": cv_rmse, "train_time_sec": train_time,
                    })

                    fig = plot_predictions(y_test.values, preds, run_name)
                    mlflow.log_figure(fig, "plots/predictions.png")
                    plt.close(fig)

                    importances = feature_importance(model, X.columns)
                    if importances is not None:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        importances.plot.barh(ax=ax)
                        ax.set_title(f"Feature importance: {model_name}")
                        fig.tight_layout()
                        mlflow.log_figure(fig, "plots/feature_importance.png")
                        plt.close(fig)

                    signature = infer_signature(X_train, model.predict(X_train))
                    # MLflow 3 serializes with skops; tree internals must be explicitly
                    # trusted. Safe here because we trained these models ourselves.
                    mlflow.sklearn.log_model(
                        model, name="model", signature=signature, input_example=X_train.head(3),
                        skops_trusted_types=TRUSTED_TREE_TYPES,
                    )

                    print(f"{run_name:<45} RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}  CV_RMSE={cv_rmse:.4f}")
                    results.append({"run_id": child.info.run_id, "run_name": run_name,
                                    "rmse": rmse, "mae": mae, "r2": r2, "cv_rmse": cv_rmse})

        leaderboard = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
        mlflow.log_table(leaderboard, "leaderboard.json")
        best = leaderboard.iloc[0]
        mlflow.log_metrics({"best_rmse": best["rmse"], "best_r2": best["r2"]})
        mlflow.set_tag("best_run", best["run_name"])

    print("\nLeaderboard (sorted by test RMSE):")
    print(leaderboard[["run_name", "rmse", "mae", "r2", "cv_rmse"]].to_string(index=False))

    register_champion(best)


def register_champion(best):
    version = mlflow.register_model(f"runs:/{best['run_id']}/model", REGISTERED_MODEL_NAME)
    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", version.version)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "run_name", best["run_name"])
    print(f"\nRegistered {REGISTERED_MODEL_NAME} v{version.version} "
          f"({best['run_name']}, RMSE={best['rmse']:.4f}) with alias 'champion'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-frac", type=float, default=1.0,
                        help="Fraction of the dataset to use (e.g. 0.2 for a quick run)")
    run_experiment(parser.parse_args().sample_frac)
