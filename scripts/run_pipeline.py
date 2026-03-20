import warnings
import sys
from pathlib import Path

# Đảm bảo src/ nằm trong sys.path khi chạy script từ scripts/ folder
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from src.data.loader import load_data
from src.data.cleaner import DataCleaner
from src.features.builder import FeatureBuilder
from src.mining.association import run_association
from src.mining.clustering import run_clustering, cluster_profile
from src.models.supervised import train_models
from src.evaluation.metrics import evaluate


def main():
    print("Running pipeline...")

    df, config = load_data()

    cleaner = DataCleaner()
    df_clean = cleaner.fit_transform(df, target_col=config.get('target', 'y'))

    builder = FeatureBuilder()
    df_feat = builder.transform(df_clean)

    # Association rules
    rules = run_association(df, cat_cols=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y'],
                             min_support=0.05, min_lift=1.2)
    print("Association rules sample:")
    print(rules.head(10))

    # Clustering
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    num_cols = [c for c in df_feat.columns if c not in cat_cols + [config['target']]]
    labels, model, silhouette = run_clustering(df_feat[num_cols], k=4)
    print(f"Silhouette score for k=4: {silhouette:.4f}")

    df_cluster = cluster_profile(df_feat[num_cols], labels)
    print(df_cluster)

    # Supervised modeling
    X = df_feat.drop(config['target'], axis=1)
    y = df_feat[config['target']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.get('test_size', 0.2), stratify=y,
                                                        random_state=config.get('random_state', 42))

    models = train_models(X_train, y_train, params=config.get('model'))

    for name, m in models.items():
        y_pred = m.predict(X_test)
        y_prob = m.predict_proba(X_test)[:,1]
        f1, pr_auc = evaluate(y_test, y_pred, y_prob)
        print(f"{name} -> F1: {f1:.4f}, PR-AUC: {pr_auc:.4f}")

    print("Done!")


if __name__ == '__main__':
    main()