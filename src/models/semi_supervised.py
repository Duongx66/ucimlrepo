import numpy as np
from sklearn.semi_supervised import LabelPropagation


def self_training(model, X_labeled, y_labeled, X_unlabeled, threshold=0.9, max_iter=3):
    X_labeled = np.asarray(X_labeled)
    y_labeled = np.asarray(y_labeled)
    X_unlabeled = np.asarray(X_unlabeled)

    model.fit(X_labeled, y_labeled)

    for _ in range(max_iter):
        probs = model.predict_proba(X_unlabeled)
        max_prob = np.max(probs, axis=1)
        pseudo_labels = np.argmax(probs, axis=1)

        selected = max_prob >= threshold
        if not np.any(selected):
            break

        X_selected = X_unlabeled[selected]
        y_selected = pseudo_labels[selected]

        X_labeled = np.vstack([X_labeled, X_selected])
        y_labeled = np.hstack([y_labeled, y_selected])

        mask = ~selected
        X_unlabeled = X_unlabeled[mask]

        model.fit(X_labeled, y_labeled)

    return model


def label_propagation(X, y):
    lp = LabelPropagation()
    lp.fit(X, y)
    return lp