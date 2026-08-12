README.md
.gitignore
requirements.txt
dvc.yaml
configs/
  train_config.yaml
src/
  data/
    dataset.py
  preprocess.py
  models/
    train.py
  evaluate.py
  monitoring/
    monitor_drift.py
  utils/
    mlflow_utils.py
tests/
  unit/
    test_preprocess.py
  data/
    test_data_validation.py
  model/
    test_model_validation.py
.github/
  workflows/
    ci.yml
reports/
  drift_report.html
compare_experiments.py
train:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Train and validate model
        run: python -m src.train --config configs/ci_config.yaml