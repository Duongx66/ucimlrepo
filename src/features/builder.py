import pandas as pd


class FeatureBuilder:
    def transform(self, df):
        df = df.copy()

        df['age_group'] = (df['age'] // 10).astype(int)
        df['balance_level'] = pd.cut(df['balance'], bins=[-1, 0, 1000, 5000, float('inf')], labels=['negative','low','medium','high'])
        df['duration_category'] = pd.cut(df['duration'], bins=[-1, 60, 180, 300, float('inf')], labels=['short','medium','long','very_long'])
        df['is_new_customer'] = (df['previous'] == 0).astype(int)

        # Chuyển categorical mới thành one-hot nếu có nhu cầu
        df = pd.get_dummies(df, columns=['balance_level','duration_category'], drop_first=True)

        return df