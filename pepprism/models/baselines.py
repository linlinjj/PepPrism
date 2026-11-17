"""Classical machine learning baseline models on AAC+DPC+PseAAC+Biophysical features."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASSICAL_MODEL_NAMES: tuple[str, ...] = (
    "LogisticRegression",
    "RandomForest",
    "SVM_RBF",
    "HistGradientBoosting",
)


def build_classical_pipeline(
    name: str,
    seed: int = 42,
    class_weight: str | None = "balanced",
    n_samples: int = 0,
) -> Pipeline:
    """Construct a scikit-learn Pipeline for a named classical baseline."""
    if name == "LogisticRegression":
        clf = LogisticRegression(
            max_iter=1500,
            C=1.0,
            class_weight=class_weight,
            random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    if name == "RandomForest":
        large = n_samples > 25000
        clf = RandomForestClassifier(
            n_estimators=160 if large else 300,
            max_depth=24 if large else None,
            min_samples_leaf=2,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    if name == "SVM_RBF":
        if n_samples > 15000:
            # Scalable RBF kernel approximation for proteome-scale imbalanced datasets
            return Pipeline([
                ("scaler", StandardScaler()),
                ("nystroem", Nystroem(kernel="rbf", n_components=300, random_state=seed)),
                ("clf", LogisticRegression(max_iter=1200, C=1.0, class_weight=class_weight, random_state=seed)),
            ])
        clf = SVC(
            C=2.0,
            kernel="rbf",
            probability=True,
            class_weight=class_weight,
            random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    if name == "HistGradientBoosting":
        clf = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            class_weight=class_weight,
            random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    raise ValueError(f"Unknown classical baseline '{name}'.")


def train_evaluate_classical_baseline(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int = 42,
    class_weight: str | None = "balanced",
) -> tuple[Pipeline, np.ndarray]:
    """Fit classical model on training features and return fitted pipeline and test probabilities."""
    pipe = build_classical_pipeline(
        name, seed=seed, class_weight=class_weight, n_samples=len(x_train)
    )
    pipe.fit(x_train, y_train)
    probs = pipe.predict_proba(x_test)[:, 1].astype(np.float64)
    return pipe, probs
