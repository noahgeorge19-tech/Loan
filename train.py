import json
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score


CSV_PATH = "loan.csv"
MODEL_OUT = "model.pkl"
METRICS_OUT = "metrics.json"


def load_data(path=CSV_PATH):
    df = pd.read_csv(path)
    return df


def build_and_train(df):
    # target
    target = "Loan Approved"

    # fix Dependents '3+'
    df["Dependents"] = df["Dependents"].replace({"3+": "3"}).astype(float)
    df[target] = df[target].astype(int)

    X = df.drop(columns=[target])
    y = df[target]

    numeric_features = [
        "Age",
        "Applicant Income($)",
        "Coapplicant Income($)",
        "Loan Amount($000)",
        "Loan Term(months)",
        "Credit History",
        "Credit Score",
        "Employment Years",
        "Existing Loans",
    ]

    categorical_features = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self Employed",
        "Property Area",
        "Home Ownership",
        "Loan Purpose",
    ]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    results = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    best_score = -1
    best_pipeline = None
    best_name = None

    for name, clf in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        results[name] = {"accuracy": acc}
        print(f"{name}: accuracy={acc:.4f}")

        if acc > best_score:
            best_score = acc
            best_pipeline = pipe
            best_name = name

    # save best pipeline and metrics
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(best_pipeline, f)
    with open(METRICS_OUT, "w") as f:
        json.dump({"best_model": best_name, "best_score": best_score, "models": results}, f, indent=2)

    print(f"Saved best model: {best_name} (accuracy={best_score:.4f}) -> {MODEL_OUT}")


def main():
    df = load_data()
    build_and_train(df)


if __name__ == "__main__":
    main()
