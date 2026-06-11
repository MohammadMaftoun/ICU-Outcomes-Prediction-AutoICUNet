# evaluation.py

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    cohen_kappa_score
)

from imblearn.metrics import (
    geometric_mean_score,
    specificity_score
)


def evaluate_classification(y_true, y_pred, y_prob):

    results = {
        "Accuracy":
            accuracy_score(y_true, y_pred) * 100,

        "Precision":
            precision_score(y_true, y_pred) * 100,

        "Recall":
            recall_score(y_true, y_pred) * 100,

        "F1":
            f1_score(y_true, y_pred) * 100,

        "Specificity":
            specificity_score(y_true, y_pred) * 100,

        "G-Mean":
            geometric_mean_score(y_true, y_pred) * 100,

        "ROC-AUC":
            roc_auc_score(y_true, y_prob) * 100,

        "MCC":
            matthews_corrcoef(y_true, y_pred) * 100,

        "Kappa":
            cohen_kappa_score(y_true, y_pred) * 100,
    }

    return results
