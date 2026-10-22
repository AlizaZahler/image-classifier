import numpy as np
from .data import CLASSES

def classification_metrics(targets, predictions):
    cm = np.zeros((3, 3), dtype=np.int64)
    for a, b in zip(targets, predictions):
        cm[int(a), int(b)] += 1
    support, predicted = cm.sum(1), cm.sum(0)
    precision = np.divide(cm.diagonal(), predicted, out=np.zeros(3), where=predicted != 0)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(3), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=precision+recall != 0)
    return {'count': int(cm.sum()), 'accuracy': float(cm.trace()/max(1, cm.sum())),
            'balanced_accuracy': float(recall[support > 0].mean()) if support.any() else 0.,
            'macro_f1': float(f1.mean()), 'confusion_matrix': cm.tolist(),
            'per_class': {label: {'precision': float(precision[i]), 'recall': float(recall[i]),
                                  'f1': float(f1[i]), 'count': int(support[i])} for label, i in CLASSES.items()}}
