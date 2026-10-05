import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

PRICE_COLS = [f"PriceIndex{i}" for i in range(1, 10)]


def engineer(df):
    df = df.copy()
    if "Junk" in df:
        df = df.drop(columns=["Junk"])

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["pipeline"]
    threshold = bundle["threshold"]

    raw = pd.read_csv(args.input)
    X = engineer(raw)
    prob = model.predict_proba(X)[:, 1]

    result = raw.copy()
    result["junk_probability"] = prob
    result["junk_prediction"] = (prob >= threshold).astype(int)
    result.to_csv(args.output, index=False)

    print(f"Saved predictions to {args.output}")
    print(f"Decision threshold: {threshold:.2f}")


if __name__ == "__main__":
    main()
