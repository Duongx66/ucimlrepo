from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def run_clustering(X, k=3):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels) if len(set(labels)) > 1 else None
    return labels, model, score


def cluster_profile(df, labels):
    df = df.copy()
    df['cluster'] = labels
    return df.groupby('cluster').agg(['mean', 'count'])