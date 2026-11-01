# MLflow Experiment Tracking Lab: California Housing

[![train](https://github.com/karunprem45/mlflow-experiment-tracking-lab/actions/workflows/train.yml/badge.svg)](https://github.com/karunprem45/mlflow-experiment-tracking-lab/actions/workflows/train.yml)

My version of [Mlflow_Labs/Lab1](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Experiment_Tracking_Labs/Mlflow_Labs/Lab1) from the MLOps course repo.

The original trains one ElasticNet model on Wine Quality. I switched to California Housing, compared 5 model types, and used the Model Registry to save and load the best one.

## What I changed

| | Original | Mine |
|---|---|---|
| Dataset | Wine Quality (1,599 rows) | California Housing (20,640 rows) |
| Models | ElasticNet | Ridge, Lasso, RandomForest, GradientBoosting, HistGradientBoosting |
| Runs | 1 run per command | 12 runs in one sweep, nested under a parent run |
| Metrics | RMSE, MAE, R² | Same, plus 5-fold CV RMSE and training time |
| Artifacts | Model | Model, prediction/residual plots, feature importance, leaderboard |
| Backend | File store | SQLite, so the Model Registry works |
| Registry | Not used | Best model registered with a `champion` alias |
| Inference | `serving.py` | `predict.py` loads the `champion` model, plus a REST API with `mlflow models serve` |
| Packaging | None | `MLproject` file, so it runs with `mlflow run .` |
| CI | None | GitHub Actions trains and predicts on every push |

## Files

- `lab.ipynb`: the whole lab as a notebook, with outputs
- `train.py`: trains and logs all models, registers the best one
- `predict.py`: loads the `champion` model and predicts on the test set
- `MLproject`: entry points for `mlflow run`
- `.github/workflows/train.yml`: CI that runs training and prediction on 10% of the data

## How to run

Tested on Python 3.14 and MLflow 3.17 (macOS). CI runs on Python 3.12 (Ubuntu).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train.py        # about 2-3 min
python predict.py
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000
```

Use `python train.py --sample-frac 0.2` for a faster run on 20% of the data.

Or run it as an MLflow project:

```bash
mlflow run . --env-manager local --experiment-name california-housing-regression -P sample_frac=0.2
mlflow run . -e predict --env-manager local --experiment-name california-housing-regression
```

## Serve the model

```bash
mlflow models serve -m "models:/CaliforniaHousingRegressor@champion" -p 5001 --env-manager local
```

In another terminal:

```bash
curl localhost:5001/invocations -H "Content-Type: application/json" -d '{
  "dataframe_split": {
    "columns": ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"],
    "data": [[8.3252,41.0,6.984,1.024,322.0,2.556,37.88,-122.23],
             [2.5313,30.0,5.039,1.193,1565.0,2.68,35.14,-119.46]]
  }
}'
```

Response:

```json
{"predictions": [4.124714443495721, 0.8192676189813848]}
```

That's about $412k for the Berkeley district and $82k for the Central Valley one.

## Results

80/20 split. The target is median house value in $100k.

| Run | RMSE | MAE | R² | CV RMSE |
|---|---|---|---|---|
| **hist_gradient_boosting (lr=0.05)** | **0.4531** | **0.3010** | **0.8433** | 0.4657 |
| hist_gradient_boosting (lr=0.1) | 0.4541 | 0.3015 | 0.8427 | 0.4658 |
| random_forest (max_depth=16) | 0.5092 | 0.3307 | 0.8021 | 0.5128 |
| gradient_boosting (lr=0.1) | 0.5114 | 0.3484 | 0.8004 | 0.5073 |
| gradient_boosting (lr=0.05) | 0.5397 | 0.3691 | 0.7777 | 0.5356 |
| random_forest (max_depth=8) | 0.5855 | 0.4026 | 0.7384 | 0.5729 |
| lasso (alpha=0.01) | 0.7404 | 0.5353 | 0.5816 | 0.7240 |
| ridge (alpha=0.1–10) | ~0.7455 | ~0.5332 | ~0.576 | ~0.7205 |
| lasso (alpha=0.1) | 0.8244 | 0.6222 | 0.4814 | 0.8198 |

Tree models did much better than linear ones (R² 0.84 vs 0.58). HistGradientBoosting was best, so it's the `champion`.

## What I learned

- Logging each model as its own run made comparing them much easier. I could sort the 12 runs by RMSE in the UI instead of writing scores down.
- Nested runs keep the experiment page tidy. All the sweep runs sit under one parent run.
- The Model Registry doesn't work with the default file store. I had to use SQLite (`sqlite:///mlflow.db`) to register models locally.
- `predict.py` loads the model by its `champion` alias, not a run ID. If I train a better model, I just move the alias.
- I logged CV RMSE next to test RMSE to check the results weren't just luck from one split. They matched closely.
- Tuning Ridge's alpha barely changed anything. Switching to gradient boosting cut RMSE by about 40%.
- MLflow 3 won't save tree models unless you mark their internal types as trusted (`skops_trusted_types`).

## Screenshots

All 12 runs sorted by RMSE:

![Runs](screenshots/runs.png)

Metric charts across runs:

![Run comparison](screenshots/compare.png)

Registered model with the `champion` alias:

![Model registry](screenshots/registry.png)
