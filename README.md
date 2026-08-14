# Heart Disease Prediction MLOps Pipeline

## Project Overview

This project is a complete machine learning operations (MLOps) pipeline for predicting heart disease using a tabular heart-disease dataset.

The prediction task is a binary classification problem:

- `target = 1` indicates that heart disease is present.
- `target = 0` indicates that heart disease is not present.

This project focuses on the machine learning lifecycle surrounding the model, including data versioning, reproducible configuration, preprocessing, experiment tracking, automated testing, CI/CD, and drift monitoring.

> **Important:** This project is for educational and technical demonstration purposes only. It is not a diagnostic tool and must not be used to make real medical decisions.

## Dataset

The dataset is stored as:

```text
data/raw/heart.csv
```

It contains 1,025 observations and 14 columns:

- 13 predictive features.
- 1 binary target column named `target`.

### Feature Dictionary

| Column | Description |
|---|---|
| `age` | Age of the patient in years |
| `sex` | Biological sex, encoded numerically |
| `cp` | Chest-pain type category |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol measurement |
| `fbs` | Fasting blood sugar indicator |
| `restecg` | Resting electrocardiographic results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina indicator |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels observed |
| `thal` | Thalassemia test category |
| `target` | Heart disease target: `1` = present, `0` = not present |

The dataset is tabular and includes numeric columns along with categorical or ordinal encoded columns such as `cp`, `restecg`, `slope`, `thal`, and `sex`. The project simulates missing values in selected features during training so the pipeline can test missing-value preprocessing and validation. [112]

## Project Components

This repository includes:

- Git version control.
- DVC data versioning.
- YAML configuration for model settings and paths.
- Data preprocessing with missing-value handling.
- Scikit-learn model training and evaluation.
- MLflow experiment tracking.
- pytest unit, data, and model validation tests.
- GitHub Actions CI/CD workflow.
- Evidently data drift monitoring.
- HTML drift-report generation.

## Repository Structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   └── train_config.yaml
├── data/
│   └── raw/
│       └── heart.csv.dvc
├── models/
├── reports/
│   └── drift_report.html
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── process_data.py
│   ├── evaluate.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── train.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── monitor_drift.py
│   └── utils/
│       ├── __init__.py
│       └── mlflow_utils.py
├── tests/
│   ├── unit/
│   │   └── test_preprocess.py
│   ├── data/
│   │   └── test_data_validation.py
│   └── model/
│       └── test_model_validation.py
├── .gitignore
├── compare_experiments.py
├── dvc.yaml
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/malissaramnauth-coder/Implement-a-Complete-MLOps-Pipeline-Project.git
```

Move into the repository:

```bash
cd Implement-a-Complete-MLOps-Pipeline-Project
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## DVC Data Versioning

The dataset is tracked using DVC instead of being committed directly to Git.

Initialize DVC:

```bash
dvc init
```

Track the heart-disease dataset:

```bash
dvc add data/raw/heart.csv
```

This creates a DVC pointer file:

```text
data/raw/heart.csv.dvc
```

Commit the DVC metadata and pointer file:

```bash
git add data/raw/heart.csv.dvc
git add .dvc/config
git add .gitignore
git commit -m "Track heart disease dataset with DVC"
```

After cloning the repository, retrieve the DVC-tracked dataset with:

```bash
dvc pull
```

DVC stores dataset metadata in Git while the actual CSV file is stored in a configured DVC remote. Running `dvc pull` retrieves the data version referenced by the committed `.dvc` pointer file. [48][56]

## Configuration

Model settings are stored in:

```text
configs/train_config.yaml
```

The YAML configuration defines:

- Dataset path.
- Target column name.
- Training and test file paths.
- Test split size.
- Random seed.
- Model type.
- Model hyperparameters.
- Minimum performance thresholds.
- MLflow experiment name.
- MLflow tracking URI.
- Drift-report path.
- Drift threshold.

Example heart-disease configuration:

```yaml
project:
  name: heart_disease_mlops
  experiment_name: heart_disease_experiments

data:
  dataset_path: data/raw/heart.csv
  train_path: data/processed/train.csv
  test_path: data/processed/test.csv
  target_column: target
  test_size: 0.2
  random_state: 42

model:
  model_type: random_forest
  n_estimators: 200
  max_depth: 8
  min_samples_split: 4
  min_samples_leaf: 2
  class_weight: balanced

metrics:
  primary_metric: f1
  minimum_accuracy: 0.75
  minimum_f1: 0.70
  minimum_roc_auc: 0.75

mlflow:
  tracking_uri: file:./mlruns
  artifact_path: model

drift:
  report_path: reports/drift_report.html
  threshold: 0.30
  production_fraction: 0.25
  random_state: 42
```

## Run Training

Run the training pipeline from the project root:

```bash
python -m src.models.train --config configs/train_config.yaml
```

The training pipeline:

1. Loads the YAML configuration.
2. Loads `data/raw/heart.csv`.
3. Splits data into training and test subsets.
4. Separates the 13 features from the `target` column.
5. Handles missing values.
6. Preprocesses numeric and encoded categorical features.
7. Trains the selected classification model.
8. Calculates evaluation metrics.
9. Saves the trained model.
10. Logs the model, configuration values, data version, and metrics to MLflow.

## Model Evaluation

The project evaluates classification performance with:

- Accuracy.
- F1 score.
- Precision.
- Recall.
- ROC-AUC.

The primary metric is F1 score because it balances precision and recall. This is useful when both false positives and false negatives matter.

The training job fails if configured performance thresholds are not met.

## MLflow Experiment Tracking

MLflow tracks every training run.

Each run logs:

- Model type.
- Number of estimators.
- Maximum tree depth.
- Minimum samples split.
- Minimum samples leaf.
- Class weight configuration.
- Dataset version hash.
- Accuracy.
- F1 score.
- Precision.
- Recall.
- ROC-AUC.
- The trained scikit-learn model pipeline.

The project logs the fitted model using:

```python
mlflow.sklearn.log_model()
```

MLflow supports logging scikit-learn models and their associated parameters, metrics, and artifacts. [89][90]

Run at least five experiments:

```bash
python run_experiments.py
```

Compare MLflow experiments by F1 score:

```bash
python compare_experiments.py \
  --experiment-name heart_disease_experiments \
  --metric f1
```

The comparison script uses `mlflow.search_runs()` to retrieve experiment results and identify the best run.

Launch the MLflow user interface:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open this address in a browser:

```text
http://127.0.0.1:5000
```

## Run Tests

Run the complete pytest suite:

```bash
pytest tests/ -v
```

### Unit Tests

Located in:

```text
tests/unit/test_preprocess.py
```

These tests verify that preprocessing functions:

- Handle missing values.
- Preserve the original DataFrame.
- Process numeric values correctly.
- Handle encoded categorical features correctly.
- Return expected feature columns.
- Raise errors for invalid input.

### Data Validation Tests

Located in:

```text
tests/data/test_data_validation.py
```

These tests verify that:

- All required heart-disease columns exist.
- The `target` column contains only `0` and `1`.
- `age` values are in a realistic range.
- `trestbps` values are positive.
- `chol` values are positive.
- `thalach` values are positive.

### Model Validation Tests

Located in:

```text
tests/model/test_model_validation.py
```

These tests verify that:

- The trained model produces predictions with the correct shape.
- Predictions contain only `0` and `1`.
- The model reaches a minimum performance threshold on a known test sample.

## Run Drift Monitoring

Run monitoring from the repository root:

```bash
python -m src.monitoring.monitor_drift \
  --config configs/train_config.yaml
```

The monitoring script:

1. Uses training data as the reference distribution.
2. Creates a simulated production dataset.
3. Runs Evidently drift detection on all input features.
4. Prints the overall drift share.
5. Lists features identified as drifted.
6. Saves an HTML report.

The report is saved here:

```text
reports/drift_report.html
```

Evidently detects data drift by comparing the distributions of values in reference and current datasets. [3][18]

## Drift Analysis

### Simulated Drifted Features

The simulated production data intentionally changes these features:

- `age`
- `trestbps`
- `chol`
- `thalach`
- `oldpeak`

### Why These Features Drifted

`age` may drift if the patient population changes, such as when a hospital begins treating a different age group.

`trestbps` may drift because the production population has different resting blood-pressure characteristics or because clinical measurement procedures change.

`chol` may drift due to population dietary patterns, medication use, laboratory calibration, or changes in patient demographics.

`thalach` may drift if exercise-testing protocols change or if the incoming patient population has a different cardiovascular fitness profile.

`oldpeak` may drift because it is related to exercise-induced ST depression and can change when patient health profiles, test protocols, or measurements change.

### Likely Impact on Model Performance

These features may affect model performance because they are clinically relevant variables in the heart-disease prediction dataset.

If the incoming production data has substantially different distributions from the data used for training, the model may become less reliable. For example:

- A shift in `age` could change the base rate of heart disease.
- A shift in `trestbps` or `chol` could change the relationship between cardiovascular risk factors and the target.
- A shift in `thalach` or `oldpeak` could change the model’s learned patterns related to exercise-test results.

This can reduce metrics such as F1 score, precision, recall, and ROC-AUC.

### Recommended Action

The recommended action is:

1. Investigate whether the detected drift is caused by a real change in the patient population, data collection process, or measurement process.
2. Continue monitoring when drift is small and model performance remains stable.
3. Validate model performance using newly labeled production data.
4. Retrain the model when drift exceeds the configured threshold or when performance declines.

Because this is a health-related classification project, drift should be investigated carefully before relying on predictions from a model trained on older data.

## CI/CD Pipeline

The GitHub Actions workflow is located at:

```text
.github/workflows/ci.yml
```

The workflow runs when:

- Code is pushed to the `main` branch.
- A pull request targets the `main` branch.

The pipeline includes two jobs.

### Test Job

The test job:

1. Checks out the repository.
2. Installs dependencies.
3. Pulls DVC-tracked data.
4. Runs all tests.

```bash
pytest tests/ -v
```

### Train Job

The train job runs only after tests pass.

The train job:

1. Installs dependencies.
2. Pulls DVC-tracked data.
3. Runs the training pipeline.
4. Fails if the model does not meet required thresholds.

```bash
python -m src.models.train --config configs/train_config.yaml
```

## Reproduce the Project

```bash
git clone https://github.com/malissaramnauth-coder/Implement-a-Complete-MLOps-Pipeline-Project.git

cd Implement-a-Complete-MLOps-Pipeline-Project

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

dvc pull

pytest tests/ -v

python -m src.models.train --config configs/train_config.yaml

python compare_experiments.py \
  --experiment-name heart_disease_experiments \
  --metric f1

python -m src.monitoring.monitor_drift \
  --config configs/train_config.yaml
```

## Author

Malissa Ramnauth
