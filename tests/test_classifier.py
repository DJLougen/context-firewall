"""Tests for the ML classifier."""

from pathlib import Path

from context_firewall.classifier import build_classifier, evaluate, predict, train
from context_firewall.io import make_row, write_jsonl
from context_firewall.labels import ContentType, Label


def test_build_classifier():
    """Should build a valid classifier pipeline."""
    pipeline = build_classifier()
    assert pipeline is not None
    assert hasattr(pipeline, "fit")
    assert hasattr(pipeline, "predict")


def test_train_on_synthetic_data(tmp_path: Path):
    """Should train on synthetic data and achieve reasonable accuracy."""
    # Generate small synthetic dataset
    train_rows = []
    for i in range(50):
        train_rows.append(make_row(
            role="system",
            content=f"You are a helpful assistant. Session {i}.",
            content_type=ContentType.SYSTEM,
            label=Label.CORE,
            turn=0,
            session_id=f"s{i}",
        ))
        train_rows.append(make_row(
            role="tool",
            content=f"94 passed, 2 failed in session {i}",
            content_type=ContentType.TOOL_RESULT_TEST,
            label=Label.DISTILL,
            turn=1,
            session_id=f"s{i}",
        ))
    
    eval_rows = []
    for i in range(10):
        eval_rows.append(make_row(
            role="system",
            content=f"You are a helpful assistant. Eval {i}.",
            content_type=ContentType.SYSTEM,
            label=Label.CORE,
        ))
        eval_rows.append(make_row(
            role="tool",
            content=f"94 passed, 2 failed in eval {i}",
            content_type=ContentType.TOOL_RESULT_TEST,
            label=Label.DISTILL,
        ))
    
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    model_path = tmp_path / "model.joblib"
    
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    
    # Train
    metrics = train(train_path, eval_path, model_path)
    
    assert metrics["accuracy"] > 0.5  # Should do better than random
    assert metrics["train_size"] == 100
    assert metrics["eval_size"] == 20
    assert model_path.exists()


def test_evaluate_model(tmp_path: Path):
    """Should evaluate a trained model."""
    # Generate data
    rows = []
    for i in range(30):
        rows.append(make_row(
            role="system",
            content=f"You are helpful. {i}",
            content_type=ContentType.SYSTEM,
            label=Label.CORE,
        ))
        rows.append(make_row(
            role="tool",
            content=f"94 passed in test {i}",
            content_type=ContentType.TOOL_RESULT_TEST,
            label=Label.DISTILL,
        ))
    
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    model_path = tmp_path / "model.joblib"
    
    write_jsonl(train_path, rows[:40])
    write_jsonl(eval_path, rows[40:])
    
    # Train
    train(train_path, eval_path, model_path)
    
    # Evaluate
    metrics = evaluate(model_path, eval_path)
    assert metrics["accuracy"] >= 0.0
    assert metrics["eval_size"] == 20


def test_predict_single(tmp_path: Path):
    """Should predict label for a single message."""
    # Generate data and train
    rows = []
    for i in range(30):
        rows.append(make_row(
            role="system",
            content=f"You are helpful. {i}",
            content_type=ContentType.SYSTEM,
            label=Label.CORE,
        ))
        rows.append(make_row(
            role="tool",
            content=f"94 passed in test {i}",
            content_type=ContentType.TOOL_RESULT_TEST,
            label=Label.DISTILL,
        ))
    
    train_path = tmp_path / "train.jsonl"
    model_path = tmp_path / "model.joblib"
    
    write_jsonl(train_path, rows)
    train(train_path, output_path=model_path)
    
    # Predict
    label, probas = predict(model_path, "You are a helpful assistant.", "system")
    assert label in ["core", "distill", "compact", "drop", "stale", "escalate"]


def test_firewall_with_ml_classifier(tmp_path: Path):
    """Firewall should work with ML classifier loaded."""
    from context_firewall.firewall import ContextFirewall, Message
    
    # Generate data and train
    rows = []
    for i in range(30):
        rows.append(make_row(
            role="system",
            content=f"You are helpful. {i}",
            content_type=ContentType.SYSTEM,
            label=Label.CORE,
        ))
        rows.append(make_row(
            role="tool",
            content=f"94 passed in test {i}",
            content_type=ContentType.TOOL_RESULT_TEST,
            label=Label.DISTILL,
        ))
    
    train_path = tmp_path / "train.jsonl"
    model_path = tmp_path / "model.joblib"
    
    write_jsonl(train_path, rows)
    train(train_path, output_path=model_path)
    
    # Create firewall with ML classifier
    fw = ContextFirewall(model_path=model_path)
    
    # Process a message
    result = fw.process(Message(role="system", content="You are a helpful assistant."))
    
    assert result.label is not None
    assert result.content is not None
