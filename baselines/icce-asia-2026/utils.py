"""Training and scoring helpers shared by all the baselines."""
import numpy as np
import torch

SEED = 1    # torch.manual_seed(SEED) at the start of every fold

def class_weights(targets, train_idx, n_classes):
    """Inverse-frequency weights for CrossEntropyLoss, from the training fold only.

    `counts.sum() / (n_classes * counts)` gives a rare class a large weight and a common class
    a small one. Scaling by `counts.sum() / n_classes` makes the weights average to 1 per
    training clip, so the loss keeps the magnitude the learning rate was tuned for.
    """
    counts = np.bincount(targets[train_idx], minlength=n_classes)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1))    # maximum() avoids dividing by 0
    return torch.tensor(weights, dtype=torch.float32)

def confusion_matrix(y_true, y_pred, n_classes):
    """cm[t, p] = number of clips of true class t that were predicted as class p."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)  # add 1 at every (true, pred) pair; a repeated pair is counted each time
    return cm

def scores(cm):
    """Pooled and macro accuracy from a confusion matrix."""
    pooled = np.trace(cm) / cm.sum()                        # correct clips / all clips
    per_class = cm.diagonal() / cm.sum(axis=1).clip(min=1)  # correct clips / row sum; clip() avoids dividing by 0
    return float(pooled), float(per_class.mean())
