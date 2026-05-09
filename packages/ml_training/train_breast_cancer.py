"""Train the BreastCancerXGB model and persist the artifact.

Run: `python -m packages.ml_training.train_breast_cancer`

The artifact bundle stores the fitted model plus its evaluated test metrics,
so the model class can populate `ModelMetadata.test_accuracy` honestly without
re-running training at load time.
"""

from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ARTIFACT_DIR = (
    Path(__file__).parent.parent.parent / "services" / "ml_mcp_server" / "artifacts"
)
ARTIFACT_PATH = ARTIFACT_DIR / "breast_cancer_xgb.joblib"


def main() -> None:
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "test_accuracy": acc, "test_f1": f1}, ARTIFACT_PATH)
    print(f"Trained breast_cancer_xgb: accuracy={acc:.3f}  f1={f1:.3f}")
    print(f"Saved artifact: {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
