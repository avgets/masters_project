from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def classification_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """
    accuracy, macro_f1, weighted_f1
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length, "
            f"got {len(y_true)} and {len(y_pred)}"
        )

    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def confusion_matrix_df(y_true: list[int], y_pred: list[int], class_ids: list[int]):
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length, "
            f"got {len(y_true)} and {len(y_pred)}"
        )

    cm = confusion_matrix(y_true, y_pred, labels=class_ids)

    index = [f"true_{class_id}" for class_id in class_ids]
    columns = [f"pred_{class_id}" for class_id in class_ids]

    return pd.DataFrame(cm, index=index, columns=columns)