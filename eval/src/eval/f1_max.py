"""Implements the F1Max metric."""

from sklearn.metrics import precision_recall_curve
import numpy as np

class F1Max:
    def __init__(self):
        self.y_true = []
        self.y_pred = []
        self.update_called = False

    def update(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_true.extend(y_true)
        self.y_pred.extend(y_pred)
        self.update_called = True
    
    def compute(self):
        precision, recall, _ = precision_recall_curve(y_true=self.y_true, probas_pred=self.y_pred)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        return f1_scores.max()
