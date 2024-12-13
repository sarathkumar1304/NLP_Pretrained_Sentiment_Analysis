from utils.logger import logger 
from typing import Dict, Tuple
import numpy as np
import evaluate

def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Compute evaluation metrics given predicted logits and labels.

    Args:
        eval_pred: A tuple of predicted logits and labels from a model.

    Returns:
        A dictionary of evaluation metrics including accuracy, precision, and F1 score.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Load metrics using `evaluate`
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    f1_metric = evaluate.load("f1")

    # Compute metrics
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    precision = precision_metric.compute(predictions=predictions, references=labels, average='weighted')["precision"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')["f1"]
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "F1": f1
    }
