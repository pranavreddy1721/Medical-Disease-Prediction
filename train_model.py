"""Train and evaluate an educational diabetes classification model."""
from pathlib import Path
import urllib.request
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
DATA_PATH = DATA_DIR / "diabetes.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "diabetes_model.joblib"
METRICS_PATH = ROOT / "model_metrics.txt"

DATA_URLS = [
    "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
]
COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
FEATURES = COLUMNS[:-1]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def ensure_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        try:
            df = pd.read_csv(DATA_PATH)
            if set(COLUMNS).issubset(df.columns):
                return df[COLUMNS]
        except Exception:
            pass
    errors = []
    for url in DATA_URLS:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=20) as response:
                raw = response.read()
            DATA_PATH.write_bytes(raw)
            try:
                df = pd.read_csv(DATA_PATH)
                if not set(COLUMNS).issubset(df.columns):
                    # Some mirrors provide the original headerless 9-column file.
                    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)
                if set(COLUMNS).issubset(df.columns) and len(df) > 100:
                    df[COLUMNS].to_csv(DATA_PATH, index=False)
                    print(f"Dataset downloaded to {DATA_PATH}")
                    return df[COLUMNS]
            except Exception as exc:
                errors.append(str(exc))
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(
        "Could not download the dataset. Check your internet connection and download a "
        "documented Pima Indians Diabetes CSV with columns " + ", ".join(COLUMNS) +
        f" to {DATA_PATH}. Details: {'; '.join(errors)}"
    )

def main():
    df = ensure_dataset().copy()
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Outcome"])
    df = df[df["Outcome"].isin([0, 1])]
    if len(df) < 100:
        raise ValueError("Dataset appears incomplete; expected at least 100 valid rows.")

    # In this dataset, zero values for these measurements generally represent missing entries.
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, float("nan"))
    X, y = df[FEATURES], df["Outcome"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced",
            min_samples_leaf=2
        ))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "Test rows": len(y_test),
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-score": f1_score(y_test, predictions, zero_division=0),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "zero_as_missing": ZERO_AS_MISSING}, MODEL_PATH)
    report = [
        "MEDICAL DISEASE PREDICTION - MODEL EVALUATION",
        "Dataset: Pima Indians Diabetes dataset (educational benchmark)",
        "Split: 80% training / 20% testing; random_state=42; stratified split",
        "Algorithm: Random Forest Classifier with median imputation",
        "",
    ]
    report += [f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
               for key, value in metrics.items()]
    report += ["", "Confusion matrix (rows=true labels 0/1, columns=predicted labels 0/1):",
               str(confusion_matrix(y_test, predictions)), "", "Classification report:",
               classification_report(y_test, predictions, zero_division=0)]
    METRICS_PATH.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved evaluation report: {METRICS_PATH}")

if __name__ == "__main__":
    main()
