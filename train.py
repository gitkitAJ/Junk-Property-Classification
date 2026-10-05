import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


PRICE_COLS = [f"PriceIndex{i}" for i in range(1, 10)]


def load_and_engineer(path):
    df = pd.read_csv(path)

    if "Junk" not in df.columns:
        raise ValueError("Expected target column 'Junk'.")

    df["ListDate"] = pd.to_datetime(df["ListDate"], errors="coerce")
    df["ListYear"] = df["ListDate"].dt.year
    df["ListMonth"] = df["ListDate"].dt.month
    df["ListDay"] = df["ListDate"].dt.day
    df["ListDayOfWeek"] = df["ListDate"].dt.dayofweek
    df["ListQuarter"] = df["ListDate"].dt.quarter
    df.drop(columns=["ListDate"], inplace=True)

    for col in PRICE_COLS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_pipeline(X):
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True))
    ])

    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=5,
            sparse_output=True
        ))
    ])

    prep = ColumnTransformer([
        ("numeric", numeric, num_cols),
        ("categorical", categorical, cat_cols)
    ])

    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.80,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    return Pipeline([("preprocess", prep), ("model", model)])


def threshold_table(y_true, prob):
    rows = []
    for threshold in np.arange(0.05, 0.56, 0.01):
        pred = (prob >= threshold).astype(int)
        rows.append({
            "threshold": round(float(threshold), 2),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Property_train.csv")
    parser.add_argument("--output", default=".")
    args = parser.parse_args()

    root = Path(args.output)
    model_dir = root / "models"
    report_dir = root / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_engineer(args.data)
    y = df.pop("Junk").astype(int)
    X = df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    pipe = build_pipeline(X_train)
    pipe.fit(X_train, y_train)

    prob = pipe.predict_proba(X_test)[:, 1]

    thresholds = threshold_table(y_test, prob)
    thresholds.to_csv(report_dir / "threshold_metrics.csv", index=False)

    best_idx = thresholds["f1"].idxmax()
    best_threshold = float(thresholds.loc[best_idx, "threshold"])
    pred = (prob >= best_threshold).astype(int)

    metrics = {
        "rows": int(len(df)),
        "features_before_encoding": int(X.shape[1]),
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "pr_auc_average_precision": float(average_precision_score(y_test, prob)),
        "best_f1_threshold": best_threshold,
        "precision_at_best_f1": float(precision_score(y_test, pred, zero_division=0)),
        "recall_at_best_f1": float(recall_score(y_test, pred, zero_division=0)),
        "f1_at_best_threshold": float(f1_score(y_test, pred, zero_division=0)),
    }

    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    report = classification_report(y_test, pred, digits=4, zero_division=0)
    (report_dir / "classification_report.txt").write_text(report)

    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Junk", "Junk"],
                yticklabels=["Non-Junk", "Junk"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — threshold={best_threshold:.2f}")
    plt.tight_layout()
    plt.savefig(report_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, prob)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(report_dir / "pr_curve.png", dpi=180)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "roc_curve.png", dpi=180)
    plt.close()

    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    importances = pipe.named_steps["model"].feature_importances_
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    fi.to_csv(report_dir / "feature_importance.csv", index=False)

    joblib.dump({
        "pipeline": pipe,
        "threshold": best_threshold,
        "feature_columns": X.columns.tolist()
    }, model_dir / "junk_property_xgb.joblib")

    print(json.dumps(metrics, indent=2))
    print("\nClassification report:\n", report)
    print(f"\nSaved model to {model_dir / 'junk_property_xgb.joblib'}")


if __name__ == "__main__":
    main()
