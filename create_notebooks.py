import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb1 = new_notebook(cells=[
    new_markdown_cell('# 03 - Mining and Clustering (Bank Marketing)'),
    new_code_cell('import pandas as pd\nfrom src.data.loader import load_data\nfrom src.mining.association import run_association\nfrom src.mining.clustering import run_clustering, cluster_profile\n\ndf, config = load_data()\nprint(f"Data shape: {df.shape}")\n'),
    new_code_cell('''# Preprocessing (basic)\nfrom src.data.cleaner import DataCleaner\nfrom src.features.builder import FeatureBuilder\n\ncleaner = DataCleaner()\ndf_clean = cleaner.fit_transform(df, target_col=config.get(\'target\', \'y\'))\n\nbuilder = FeatureBuilder()\ndf_feat = builder.transform(df_clean)\n\nprint(\'Completed feature builder.\n\', df_feat.head(2))\n'''),
    new_code_cell('''# Association rule mining\ncat_cols = [\'job\',\'marital\',\'education\',\'default\',\'housing\',\'loan\',\'contact\',\'month\',\'poutcome\',\'y\']\nrules = run_association(df_feat, cat_cols=cat_cols, min_support=0.05, min_lift=1.2)\nprint(\'Rules top 10:\')\nprint(rules[[\'antecedents\',\'consequents\',\'support\',\'confidence\',\'lift\']].head(10))\n'''),
    new_code_cell('''# Clustering on numerical features\nnum_cols = [c for c in df_feat.columns if c not in cat_cols]\nfrom sklearn.preprocessing import StandardScaler\nX = df_feat[num_cols].copy()\nX_scaled = StandardScaler().fit_transform(X)\nlabels, model, sil = run_clustering(X_scaled, k=4)\nprint(f\'Silhouette: {sil:.4f}\')\nprofile = cluster_profile(df_feat[num_cols], labels)\nprint(profile.head())\n''')
])
with open('notebooks/03_mining_clustering.ipynb','w',encoding='utf-8') as f:
    nbformat.write(nb1, f)

nb2 = new_notebook(cells=[
    new_markdown_cell('# 04b - Semi-supervised experiments'),
    new_code_cell('import pandas as pd\nfrom src.data.loader import load_data\nfrom src.data.cleaner import DataCleaner\nfrom src.features.builder import FeatureBuilder\nfrom sklearn.model_selection import train_test_split\nfrom src.models.supervised import train_models\nfrom src.models.semi_supervised import self_training, label_propagation\nfrom src.evaluation.metrics import evaluate\n\ndf, config = load_data()\ncleaner = DataCleaner()\ndf_clean = cleaner.fit_transform(df, target_col=config.get(\'target\',\'y\'))\nbuilder = FeatureBuilder()\ndf_feat = builder.transform(df_clean)\n\nX = df_feat.drop(config.get(\'target\',\'y\'), axis=1)\ny = df_feat[config.get(\'target\',\'y\')]\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.get(\'test_size\',0.2), stratify=y, random_state=config.get(\'random_state\',42))\n'''),
    new_code_cell('''def run_semi(label_pct):\n    n = int(len(X_train) * label_pct / 100)\n    idx = y_train.sample(frac=1, random_state=42).index\n    labeled_idx = idx[:n]\n    unlabeled_idx = idx[n:]\n\n    X_l = X_train.loc[labeled_idx]\n    y_l = y_train.loc[labeled_idx]\n    X_u = X_train.loc[unlabeled_idx]\n\n    base = train_models(X_l, y_l, params=config.get(\'model\'))['rf']\n    semi = self_training(base, X_l, y_l, X_u, threshold=0.9, max_iter=3)\n    y_pred = semi.predict(X_test)\n    y_prob = semi.predict_proba(X_test)[:,1]\n    f1, pr_auc = evaluate(y_test, y_pred, y_prob)\n    return f1, pr_auc\n\nfor p in [5,10,20]:\n    f1, pr_auc = run_semi(p)\n    print(f\'\\nLabeled %: {p}, F1={f1:.4f}, PR-AUC={pr_auc:.4f}\')\n'''),
    new_code_cell('''from sklearn.semi_supervised import LabelPropagation\n\ny_semi = y_train.copy()\ny_semi.iloc[int(len(y_train)*0.5):] = -1\nmodel_lp = LabelPropagation()\nmodel_lp.fit(X_train, y_semi)\ny_pred = model_lp.predict(X_test)\ny_prob = model_lp.predict_proba(X_test)[:,1]\nprint('LabelPropagation', evaluate(y_test, y_pred, y_prob))\n''')
])
with open('notebooks/04b_semi_supervised.ipynb','w',encoding='utf-8') as f:
    nbformat.write(nb2, f)

print('Notebooks created successfully')
