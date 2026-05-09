"""Train the DiabetesCATBoost model and persist the artifact.

Run: `python -m packages.ml_training.train_diabetes`

Dataset: Pima Indians Diabetes Database (UCI / OpenML 'diabetes' v1, n=768).
Model:   CatBoostClassifier, binary task (tested_positive vs tested_negative).

The artifact bundle stores the fitted model, evaluated test metrics, the
ordered feature names, and one representative sample row — so the model
class can run, self-describe, and provide `sample_inputs()` without ever
re-fetching the dataset.
"""

from pathlib import Path

import joblib
from catboost import CatBoostClassifier
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ARTIFACT_DIR = (
    Path(__file__).parent.parent.parent / "services" / "ml_mcp_server" / "artifacts"
)
ARTIFACT_PATH = ARTIFACT_DIR / "diabetes_catboost.joblib"


def main() -> None:
    bunch = fetch_openml(name="diabetes", version=1, as_frame=True, parser="auto")
    X_df = bunch.data
    # OpenML target is a string label: "tested_positive" / "tested_negative".
    y = (bunch.target == "tested_positive").astype(int).values
    feature_names = list(X_df.columns)

    X = X_df.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = CatBoostClassifier(
        iterations=300,
        depth=4,
        learning_rate=0.05,
        loss_function="Logloss",
        verbose=False,
        random_seed=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))

    sample = {name: float(X_df.iloc[0][name]) for name in feature_names}

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "test_accuracy": acc,
            "test_f1": f1,
            "feature_names": feature_names,
            "sample_inputs": sample,
        },
        ARTIFACT_PATH,
    )
    print(f"Trained diabetes_catboost: accuracy={acc:.3f}  f1={f1:.3f}")
    print(f"Saved artifact: {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
