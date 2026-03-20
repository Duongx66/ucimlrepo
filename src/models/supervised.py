from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_models(X_train, y_train, params=None):
    params = params or {}

    logreg_params = params.get('logistic', {})
    rf_params = params.get('rf', {})
    xgb_params = params.get('xgb', {})

    models = {
        "logreg": LogisticRegression(max_iter=1000, **logreg_params),
        "rf": RandomForestClassifier(random_state=42, **rf_params),
        "xgb": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, **xgb_params)
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models