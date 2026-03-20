from sklearn.metrics import f1_score, precision_recall_curve, auc

def evaluate(y_true, y_pred, y_prob):
    f1 = f1_score(y_true, y_pred)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    return f1, pr_auc