# Fake News Detection

A small Flask web application for classifying submitted news text as `FAKE` or `REAL`. The web page accepts article text and displays the label returned by a saved scikit-learn classifier.

> **Model and data note:** The repository includes `model.pkl`, which is loaded as the classifier. The application also requires a local `news.csv` file with `text` and `label` columns to fit its TF-IDF vectorizer when the app starts and on each prediction. No dataset is included here, so the app cannot run until that file is supplied. The committed files also place `index.html` and `style.css` at the project root, while Flask expects them under `templates/` and `static/` respectively; place them in those Flask folders to use the page as written.

## Features

- Browser form for entering news text.
- Flask `POST /predict` route that returns a prediction on the same page.
- Displays `FAKE` results in red and `REAL` results in green.
- Saved classifier loaded from `model.pkl`.
- Separate training script (`model.py`) that trains and serializes a classifier from `news.csv`.

## Technology

- Python
- Flask and Jinja2
- pandas
- scikit-learn: `TfidfVectorizer`, `PassiveAggressiveClassifier`, train/test split, and evaluation helpers
- HTML and CSS
- Pickle for saving/loading the classifier

Dependency versions are pinned in [`requirements.txt`](requirements.txt).

## Project structure

```text
.
├── app.py             # Flask application and prediction route
├── model.py           # Model training, evaluation, and pickle export
├── model.pkl          # Serialized classifier consumed by app.py
├── index.html         # Prediction form template (Flask expects templates/index.html)
├── style.css          # Page stylesheet (Flask expects static/style.css)
├── requirements.txt   # Python dependencies
└── news.csv           # Required local input; not included
```

For the current Flask template/static references, use this layout when preparing a runnable checkout:

```text
templates/index.html
static/style.css
```

## Dataset input expected by the code

Both Python scripts read `news.csv` from the process working directory. The code accesses these columns:

| Column | Purpose |
| --- | --- |
| `text` | News text used as the model input |
| `label` | Target class used for training and displayed predictions (`FAKE` or `REAL`) |

The project does not document or include the dataset source, row count, or a verified model score. Supply a CSV compatible with the code; no dataset details or performance claims are assumed here.

## Installation

Use a Python environment compatible with the pinned dependencies (the requirements include older versions of NumPy, pandas, SciPy, and scikit-learn).

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Before starting the web app, make sure the project directory contains `news.csv` and arrange the HTML and CSS files in `templates/` and `static/`, as shown above.

## Run the application

From the project directory, with the environment activated and required files in place:

```bash
python app.py
```

The script starts Flask with debug mode enabled. Open the local address printed by Flask (normally `http://127.0.0.1:5000/`), enter news text, and submit the form. Debug mode is suitable only for local development.

## Model workflow

### Training and export

`model.py` reads `news.csv`, selects `text` and `label`, and splits the rows into training and test partitions with `test_size=0.2` and `random_state=0`. It fits a `TfidfVectorizer` configured with English stop words and `max_df=0.7`, then fits a `PassiveAggressiveClassifier` with `max_iter=50`. It prints an accuracy score and confusion matrix for that split and pickles the fitted classifier to `model.pkl`.

### Web inference

`app.py` loads `model.pkl` and reads the same CSV at startup. For a prediction, it fits a TF-IDF vectorizer on the training text split, transforms the submitted text, and passes that vector to the loaded classifier. The predicted class is rendered in `index.html`.

The vectorizer itself is not serialized with the classifier. The app reconstructs it from `news.csv`; consequently, the CSV and training split must be compatible with the training data used to create `model.pkl`. The test split is transformed in the request function but is not used for the prediction. Training code prints an evaluation score when run; no specific result is claimed in this README.

To train and write a fresh `model.pkl` after providing `news.csv`:

```bash
python model.py
```

## Future improvements

- Package the fitted vectorizer and classifier together so inference uses the exact preprocessing used during training.
- Avoid fitting the vectorizer again for every request; fit/load it once when the application starts.
- Add validation and a helpful response for blank or malformed submissions and missing data/model files.
- Add automated tests for training, prediction, and the Flask routes.
- Move dataset paths and runtime settings into configuration, and disable Flask debug mode outside local development.
- Document the dataset provenance and measured evaluation results once those have been established and verified.
