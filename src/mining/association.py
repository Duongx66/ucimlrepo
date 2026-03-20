import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules


def run_association(df, cat_cols=None, min_support=0.05, min_lift=1.2):
    if cat_cols is None:
        cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']

    df_cat = df[cat_cols].astype(str).copy()
    df_cat = pd.get_dummies(df_cat, drop_first=False)
    df_cat = df_cat.clip(0, 1).astype('int8')

    # Memory guard: keep only top frequent item columns when quá lớn
    if df_cat.shape[1] > 2000:
        top_cols = df_cat.sum().sort_values(ascending=False).head(2000).index
        df_cat = df_cat[top_cols]

    if df_cat.shape[1] > 3000:
        raise MemoryError('Too many one-hot columns in association mining; reduce cat_cols or increase min_support')

    freq = fpgrowth(df_cat, min_support=min_support, use_colnames=True)
    rules = association_rules(freq, metric='lift', min_threshold=min_lift)

    return rules.sort_values(by='lift', ascending=False)

