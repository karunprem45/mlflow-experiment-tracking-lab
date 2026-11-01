"""Load the champion model from the MLflow Model Registry and run batch predictions."""

import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import mlflow

from train import REGISTERED_MODEL_NAME, SEED, TRACKING_URI, eval_metrics


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Loaded {model_uri}")

    data = fetch_california_housing(as_frame=True)
    X, y = data.frame.drop(columns=["MedHouseVal"]), data.frame["MedHouseVal"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    preds = model.predict(X_test)
    rmse, mae, r2 = eval_metrics(y_test, preds)
    print(f"Batch inference on {len(X_test)} rows: RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}\n")

    sample = X_test.head(5).copy()
    sample["actual_$"] = (y_test.head(5) * 100_000).round(0)
    sample["predicted_$"] = (np.asarray(preds[:5]) * 100_000).round(0)
    print(sample[["MedInc", "HouseAge", "AveRooms", "Latitude", "Longitude", "actual_$", "predicted_$"]]
          .to_string())


if __name__ == "__main__":
    main()
