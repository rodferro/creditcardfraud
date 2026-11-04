# Credit card fraud detection

This project scores credit card transactions by how likely they are to be fraudulent, using the [Kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284k transactions, 0.17% fraud).

**Results in brief:** on a held-out later time period, a strongly regularised logistic regression and a tuned LightGBM reach the same average precision, about 0.80, against a no-skill baseline of 0.0013. At a cost-optimal threshold, either model catches about three-quarters of the frauds with 8–10 alerts per hour.

**[Read the report →](REPORT.md)** · [Slides (PDF)](slides.pdf)

## Repository layout

```
├── README.md            this file
├── REPORT.md            results write-up (start here)
├── slides.pdf           presentation deck (14 slides)
├── creditcard.ipynb     the full analysis, executed with outputs
├── fraud_utils.py       transformers, evaluation helpers and plot style used by the notebook
├── figures/             charts saved by the notebook and embedded in REPORT.md
├── requirements.txt     pinned dependencies
└── data/                dataset goes here (git-ignored)
```

## Setup

The project was tested with **Python 3.12** from Homebrew on macOS (arm64).

```bash
brew install python@3.12 libomp        # libomp is needed by LightGBM on macOS
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Get the data

The dataset (~150 MB) is not committed. You can download it with the Kaggle CLI. This needs an API token in `~/.kaggle/`; see the [Kaggle docs](https://www.kaggle.com/docs/api#authentication).

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip
```

Alternatively, download `creditcard.csv` from the dataset page by hand and place it at `data/creditcard.csv`.

## Running the analysis

Open the notebook interactively:

```bash
jupyter lab creditcard.ipynb
```

Or re-execute it end to end, which also regenerates `figures/`:

```bash
jupyter nbconvert --to notebook --execute --inplace creditcard.ipynb
```

A full run takes about **6 minutes** on a 10-core laptop. Most of that time is the LightGBM hyperparameter search. All random seeds are fixed, so re-runs reproduce the numbers in the report.

## Approach in one paragraph

Duplicates are removed. The data is split by time: train on the first 80% of the timeline, test once on the last 20%. Every modelling choice is made with forward-chaining cross-validation inside the training period, scored by average precision. The candidates are:

- a logistic regression, tested across regularisation strengths and rebalancing strategies (class weights, SMOTE, undersampling, an Isolation Forest anomaly feature);
- a LightGBM, tuned by random search.

Alert thresholds are chosen on out-of-fold training predictions with an explicit cost model (€5 per review, transaction amount per missed fraud), then applied to the test period. See [REPORT.md](REPORT.md) for findings and caveats.
