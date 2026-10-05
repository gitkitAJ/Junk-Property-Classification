# Junk Property Classification — Industry-Level Data Science Project

## Business Objective
Build an end-to-end machine-learning system that classifies properties as **Junk (1)** or **Non-Junk (0)** from listing, pricing, agency, property, geographic and market attributes.

The dataset contains 62,035 records and the target is imbalanced: approximately 87.7% Non-Junk and 12.3% Junk. Therefore, the project evaluates ROC-AUC, PR-AUC, precision, recall and F1 rather than relying on accuracy alone.

## Project Workflow
1. Data quality and target analysis
2. Date feature engineering from `ListDate`
3. Numeric conversion for `PriceIndex*` fields
4. Missing-value handling
5. High-cardinality categorical encoding with one-hot encoding and minimum-frequency grouping
6. Stratified train/validation split
7. XGBoost binary classification
8. Probability-threshold optimization
9. Model evaluation and confusion matrix
10. Feature importance extraction
11. Saved model for batch/inference use

## Dataset
Expected file:
`Property_train.csv`

Target:
`Junk`

Important feature groups:
- Pricing: `PriceIndex1` ... `PriceIndex9`
- Property: `Material`, `Architecture`, `SubModel`, `Facade`, `InteriorsStyle`, `PropertyAge`, `BuildYear`
- Location: `State`, `Zip`, `Region`, `RegionType`, `NormalisedPopulation`
- Listing/agency: `ListDate`, `Agency`, `Channel`, `ExpeditedListing`
- Other: `InsurancePremiumIndex`, `AreaIncomeType`, `EnvRating`, `PRIMEUNIT`, `PlotType`

## Why this is a classification problem
`Junk` is a binary target. The model outputs a probability that a property belongs to the Junk class. A business-selected threshold converts the probability into a final class.

## Model
The reference implementation uses:
- `ColumnTransformer`
- `SimpleImputer`
- `OneHotEncoder(handle_unknown="ignore", min_frequency=5)`
- `XGBClassifier`

The preprocessing is kept inside a single sklearn `Pipeline`, which prevents train/test preprocessing leakage and makes inference reproducible.

## Evaluation
The project reports:
- ROC-AUC
- PR-AUC / Average Precision
- Precision
- Recall
- F1
- Confusion matrix
- Threshold-performance table

For this use case, recall is important because missing a Junk property may be more costly than sending an additional property for review. The threshold should therefore be selected with business costs in mind rather than automatically using 0.50.

## Run

```bash
pip install -r requirements.txt
python src/train.py --data Property_train.csv
```

Outputs:
- `models/junk_property_xgb.joblib`
- `reports/metrics.json`
- `reports/threshold_metrics.csv`
- `reports/confusion_matrix.png`
- `reports/pr_curve.png`
- `reports/roc_curve.png`
- `reports/feature_importance.csv`

For prediction:

```bash
python src/predict.py --model models/junk_property_xgb.joblib --input Property_train.csv --output reports/predictions.csv
```

## Resume-ready project statement
**Junk Property Classification | Python, Pandas, Scikit-learn, XGBoost**
- Developed an end-to-end imbalanced binary-classification pipeline to identify potentially junk properties from 62K+ listings.
- Engineered date and numeric features, handled missing/high-cardinality categorical data, and trained an XGBoost classifier using leakage-safe sklearn preprocessing.
- Evaluated model performance using ROC-AUC, PR-AUC, precision, recall and F1, with probability-threshold tuning to prioritize detection of the minority Junk class.

## Notes
This project is a property-quality classification/risk-screening model, not a legal, valuation, lending or investment decision system. Predictions should be reviewed against business rules and operational data before production use.
