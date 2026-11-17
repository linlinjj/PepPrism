"""Machine learning and deep learning models for therapeutic peptide prediction."""

from pepprism.models.baselines import (
    CLASSICAL_MODEL_NAMES,
    train_evaluate_classical_baseline,
)
from pepprism.models.pepprism_net import (
    BiGRUAttentionBaseline,
    PepPrismNet,
    SeqCNN1DBaseline,
)
from pepprism.models.trainer import (
    EvaluationMetrics,
    TrainedPepPrismBundle,
    compute_classification_metrics,
    predict_pepprism_probabilities,
    train_deep_model,
)

__all__ = [
    "BiGRUAttentionBaseline",
    "CLASSICAL_MODEL_NAMES",
    "EvaluationMetrics",
    "PepPrismNet",
    "SeqCNN1DBaseline",
    "TrainedPepPrismBundle",
    "compute_classification_metrics",
    "predict_pepprism_probabilities",
    "train_deep_model",
    "train_evaluate_classical_baseline",
]
