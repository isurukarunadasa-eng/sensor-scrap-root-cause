"""
model.py

Trains and evaluates the supervised classifier (the project's IMPROVED
method) on units with confirmed root-cause labels. Also computes SHAP
feature importance for explainability / root-cause ranking.

TODO:
    - train_model(features_df, labels) -> fitted model
    - predict_cause(model, unit_features) -> (predicted_cause, confidence)
    - explain_prediction(model, unit_features) -> ranked list of contributing
      parameters (SHAP values)
"""


def train_model(features_df, labels):
    """Train the classifier on labeled units. Not yet implemented."""
    raise NotImplementedError


def predict_cause(model, unit_features):
    """Predict root cause + confidence for one unit. Not yet implemented."""
    raise NotImplementedError
