"""CPU-only message classifier for label prediction.

Uses TF-IDF features + VotingClassifier (SGD + Naive Bayes + Logistic Regression)
with optional probability calibration. Same pattern as busyBee-cpu.

Model versioning: trained models include a _cf_version attribute for
forward compatibility checks.
"""

from __future__ import annotations

_MODEL_VERSION = "0.1.0"


def _load_model(model_path: str | Path) -> tuple[Any, str | None]:
    """Load a model, handling both old (bare pipeline) and new (versioned dict) formats.

    Returns (pipeline, version_string).
    """
    loaded = joblib.load(model_path)
    if isinstance(loaded, dict) and "pipeline" in loaded:
        return loaded["pipeline"], loaded.get("version")
    # Legacy format: bare pipeline
    return loaded, None

import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from honeycomb.features import features_to_text
from honeycomb.io import load_training_data
from honeycomb.labels import Label


def build_classifier() -> Pipeline:
    """Build the classifier pipeline.
    
    Returns a Pipeline with:
    - TF-IDF vectorizer for text features
    - VotingClassifier ensemble (SGD + NB + LR)
    - Optional calibration
    """
    # TF-IDF for text features
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    
    # Base classifiers
    sgd = SGDClassifier(
        loss="modified_huber",  # Supports predict_proba
        alpha=1e-4,
        max_iter=1000,
        random_state=42,
    )
    
    nb = MultinomialNB(alpha=0.1)
    
    lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
    )
    
    # Voting ensemble
    voting = VotingClassifier(
        estimators=[
            ("sgd", sgd),
            ("nb", nb),
            ("lr", lr),
        ],
        voting="soft",
    )
    
    # Pipeline
    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("classifier", voting),
    ])
    
    return pipeline


def train(
    train_path: str | Path,
    eval_path: str | Path | None = None,
    output_path: str | Path = "models/honeycomb.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train the classifier on labeled data.
    
    Args:
        train_path: Path to training JSONL.
        eval_path: Optional path to evaluation JSONL. If None, splits train_path.
        output_path: Where to save the trained model.
        test_size: Fraction of data to use for evaluation (if eval_path is None).
        random_state: Random seed for reproducibility.
    
    Returns:
        Training metrics dict.
    """
    # Load data
    feature_texts, labels = load_training_data(train_path)
    
    if len(feature_texts) < 10:
        raise ValueError(f"Not enough training data: {len(feature_texts)} rows")
    
    # Split if no eval set provided
    if eval_path is None:
        X_train, X_eval, y_train, y_eval = train_test_split(
            feature_texts, labels, test_size=test_size, random_state=random_state
        )
    else:
        X_train = feature_texts
        y_train = labels
        X_eval, y_eval = load_training_data(eval_path)
    
    print(f"Training on {len(X_train)} examples, evaluating on {len(X_eval)} examples")
    
    # Build and train
    pipeline = build_classifier()
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_eval)
    accuracy = accuracy_score(y_eval, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_eval, y_pred, zero_division=0))
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "version": _MODEL_VERSION}, output_path)
    print(f"Model saved to {output_path} (version {_MODEL_VERSION})")
    
    return {
        "accuracy": accuracy,
        "train_size": len(X_train),
        "eval_size": len(X_eval),
        "output_path": str(output_path),
    }


def evaluate(
    model_path: str | Path,
    eval_path: str | Path,
) -> dict[str, Any]:
    """Evaluate a trained model on a dataset.
    
    Returns evaluation metrics.
    """
    # Load model
    pipeline, version = _load_model(model_path)
    if version:
        print(f"Model version: {version}")
    
    # Load data
    feature_texts, labels = load_training_data(eval_path)
    
    # Predict
    predictions = pipeline.predict(feature_texts)
    accuracy = accuracy_score(labels, predictions)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, predictions, zero_division=0))
    
    return {
        "accuracy": accuracy,
        "eval_size": len(feature_texts),
    }


def predict(
    model_path: str | Path,
    content: str,
    role: str,
) -> tuple[str, dict[str, float]]:
    """Predict label for a single message.
    
    Returns (label, probabilities).
    """
    from honeycomb.features import extract_features
    
    # Load model
    pipeline, _ = _load_model(model_path)
    
    # Extract features
    features = extract_features(content, role)
    feature_text = features_to_text(features)
    
    # Predict
    label = pipeline.predict([feature_text])[0]
    
    # Get probabilities if available
    probabilities = {}
    if hasattr(pipeline, "predict_proba"):
        try:
            probas = pipeline.predict_proba([feature_text])[0]
            classes = pipeline.classes_
            probabilities = {cls: float(prob) for cls, prob in zip(classes, probas)}
        except Exception:
            pass
    
    return label, probabilities
