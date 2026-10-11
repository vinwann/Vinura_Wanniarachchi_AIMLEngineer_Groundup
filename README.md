# Centrifuge bearing anomaly detection

This project contains exploratory analysis and a healthy-reference Isolation Forest for centrifuge motor vibration data.

## 1. Install uv

uv manages Python and the project's dependencies. Use the installer for your operating system, following the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

**macOS / Linux:**

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal after installation, then check:

```sh
uv --version
```

## 2. Set up the project

Download or clone the project, then open a terminal in its root directory:

```sh
uv sync 
```

The project uses Python 3.12, as specified in `.python-version`. The sync command creates `.venv` and installs dependencies from `uv.lock`. There is no need to activate the environment when using `uv run`.

Place the supplied dataset in the project root with this exact filename:

```text
centrifuge line datasets.parquet
```

The dataset is approximately 500 MiB and is excluded from Git. It must be supplied separately. Keep `feature_utils.py` and `model_inferer.py` in the project root alongside the notebooks.

## 3. Start Jupyter and run the notebooks

From the project root, launch JupyterLab:

```sh
uv run --locked --with jupyterlab --with ipykernel jupyter lab
```

This makes JupyterLab and its Python kernel available for the session without changing the project's dependency files. See the [uv Jupyter guide](https://docs.astral.sh/uv/guides/integration/jupyter/).

Open the URL printed in the terminal if your browser does not open automatically. Select the Python kernel supplied by this command, then run cells in order:

1. **`Q_1_EDA.ipynb`** — exploratory analysis, operating-state inspection and frequency-band hypotheses. Use **Kernel → Restart Kernel and Run All Cells**.
2. **`Q2_model.ipynb`** — feature engineering, comparison of eight-, four- and five-feature Isolation Forest configurations, model saving and holdout inference. Start with a fresh kernel and run from the first cell through **“Run inference using the saved model.”** Q2 loads the data itself, so it does not depend on Q1's kernel state.

**Current Q2 limitation:** the later section titled **“A second model: interpretable Bearing Health Index”** imports `bhi_model.py`, which is absent from this project. Stop before that section; running the entire Q2 notebook will fail there. The Isolation Forest workflow preceding it is runnable.

Feature extraction can take some time and requires enough memory for the waveform dataset. Leave the kernel running until each cell finishes. Press **Ctrl+C** in the terminal to stop Jupyter when finished.

## 4. Outputs and inference

The Q2 Isolation Forest workflow displays the feature comparisons, evaluation metrics and holdout score plots. Its save cell writes:

```text
artifacts/bearing_anomaly_model.joblib
```

Rerunning the save cell replaces this artifact with the newly fitted model and calibrated threshold. Holdout scores are kept in the notebook variable `hold_out_scores`.

To score already engineered features with the saved model, run this in a notebook cell:

```python
from model_inferer import ModelInferer

inferer = ModelInferer("artifacts/bearing_anomaly_model.joblib")
predictions = inferer.predict(hold_out_scores[inferer.feature_columns])
display(predictions.head())
```

`ModelInferer` accepts a feature dictionary, a list of dictionaries or a DataFrame. Supply the same four engineered features and processing used during training; raw waveforms are not accepted directly. It returns `anomaly_score`, `threshold` and `anomaly`. An anomaly flag indicates departure from the healthy reference, rather than a confirmed bearing fault.

The report section is in [Q2_Model_Training_Report.md](Q2_Model_Training_Report.md), with linked figures in `report_figures/`.

## Troubleshooting

- **`uv: command not found`:** restart the terminal and follow the installer's instructions for adding uv to your PATH.
- **Dataset not found:** check the filename and launch Jupyter from the project root.
- **Missing analysis packages:** use the Jupyter command above and its Python kernel rather than an unrelated global Python environment.
- **`No module named bhi_model`:** you have reached the supplementary BHI section; stop before it to run the supplied Isolation Forest workflow.
