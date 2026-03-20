import streamlit as st
import pandas as pd
from src.data.loader import load_data
from src.data.cleaner import DataCleaner
from src.features.builder import FeatureBuilder
from src.mining.association import run_association
from src.mining.clustering import run_clustering, cluster_profile
from src.models.supervised import train_models
from src.models.semi_supervised import self_training
from src.evaluation.metrics import evaluate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@st.cache_data
def get_data():
    df, config = load_data()
    return df, config


@st.cache_data
def preprocess_data(df, config):
    cleaner = DataCleaner()
    df_clean = cleaner.fit_transform(df, target_col=config.get('target', 'y'))
    builder = FeatureBuilder()
    df_feat = builder.transform(df_clean)
    return df_feat


def main():
    st.title('Bank Marketing Data Mining App')
    st.sidebar.title('Navigation')
    page = st.sidebar.selectbox('Go to', ['EDA', 'Association Mining', 'Clustering', 'Classification', 'Semi-supervised'])

    df, config = get_data()
    df_feat = preprocess_data(df, config)

    if page == 'EDA':
        st.header('Exploratory Data Analysis')
        st.write('Sample data:')
        st.write(df.head())
        st.write('Target distribution:')
        st.write(df[config.get('target', 'y')].value_counts(normalize=True))
        st.write('Numeric summary:')
        st.write(df.describe())

    elif page == 'Association Mining':
        st.header('Association Rule Mining')
        st.write('Use categorical features only; high cardinality can consume memory.')

        cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
        min_support = st.sidebar.slider('min_support', 0.05, 0.3, 0.1, 0.01)
        min_lift = st.sidebar.slider('min_lift', 1.0, 5.0, 1.5, 0.1)

        max_rules = st.sidebar.slider('max_rules_display', 5, 50, 20, 5)

        df_for_assoc = df_feat[cat_cols].astype(str)
        st.write('Categorical distribution sample:')
        st.write(df_for_assoc.iloc[:5])

        if df_for_assoc.shape[0] > 50000:
            st.info('Dataset lớn; apriori sẽ chạy chậm, xem thông số mean support cao hơn.')

        try:
            with st.spinner('Mining rules...'):
                rules = run_association(df_feat, cat_cols=cat_cols, min_support=min_support, min_lift=min_lift)
            st.success(f'Found {len(rules)} rules')
            st.write(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(max_rules))

            if not rules.empty:
                top_by_lift = rules.sort_values('lift', ascending=False).head(max_rules)
                st.bar_chart(top_by_lift.set_index('antecedents')['lift'])

        except MemoryError:
            st.error('MemoryError: Dữ liệu lớn, hãy tăng min_support hoặc giảm số lượng cột categorical trong run_association')
        except Exception as e:
            st.error(f'Error: {e}')


    elif page == 'Clustering':
        st.header('Clustering (KMeans)')
        num_cols = [c for c in df_feat.columns if c not in ['job','marital','education','default','housing','loan','contact','month','poutcome','y']]
        k = st.sidebar.slider('n_clusters', 2, 10, 4)
        X = df_feat[num_cols]
        X_scaled = StandardScaler().fit_transform(X)
        labels, model, silhouette = run_clustering(X_scaled, k=k)
        st.write(f'Silhouette score (k={k}):', silhouette)
        profile = cluster_profile(df_feat[num_cols], labels)
        st.write(profile)

    elif page == 'Classification':
        st.header('Classification Models')
        X = df_feat.drop(config.get('target', 'y'), axis=1)
        y = df_feat[config.get('target', 'y')]
        test_size = st.sidebar.slider('test_size', 0.1, 0.4, 0.2, 0.05)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

        models = train_models(X_train, y_train, params=config.get('model'))
        records = []
        for name, m in models.items():
            y_pred = m.predict(X_test)
            y_prob = m.predict_proba(X_test)[:, 1]
            f1, pr_auc = evaluate(y_test, y_pred, y_prob)
            records.append((name, f1, pr_auc))

        result_df = pd.DataFrame(records, columns=['model', 'f1', 'pr_auc'])
        st.write(result_df)

    else:
        st.header('Semi-supervised Learning')
        X = df_feat.drop(config.get('target', 'y'), axis=1)
        y = df_feat[config.get('target', 'y')]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        label_fracs = [5, 10, 20]
        results = []

        for p in label_fracs:
            n_labeled = max(1, int(len(X_train) * p / 100))
            idx = y_train.sample(frac=1, random_state=42).index
            labeled_idx = idx[:n_labeled]
            unlabeled_idx = idx[n_labeled:]
            X_l = X_train.loc[labeled_idx]
            y_l = y_train.loc[labeled_idx]
            X_u = X_train.loc[unlabeled_idx]
            base = train_models(X_l, y_l, params=config.get('model'))['rf']

            semi = self_training(base, X_l, y_l, X_u, threshold=0.9, max_iter=3)
            y_pred = semi.predict(X_test)
            y_prob = semi.predict_proba(X_test)[:, 1]
            f1, pr_auc = evaluate(y_test, y_pred, y_prob)
            results.append((p, f1, pr_auc))

        st.write(pd.DataFrame(results, columns=['labeled_pct', 'f1', 'pr_auc']))


if __name__ == '__main__':
    main()
