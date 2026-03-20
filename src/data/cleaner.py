import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataCleaner:
    def __init__(self):
        self.encoders = {}

    def fit_transform(self, df: pd.DataFrame, target_col: str = 'y') -> pd.DataFrame:
        df = df.copy()

        # Thay thế missing giá trị nếu có
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        # Mã hóa target trước nếu là chuỗi
        if target_col in df.columns and df[target_col].dtype == object:
            le_t = LabelEncoder()
            df[target_col] = le_t.fit_transform(df[target_col])
            self.encoders[target_col] = le_t

        for col in df.select_dtypes(include='object').columns:
            if col == target_col:
                continue
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le

        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, le in self.encoders.items():
            if col in df.columns:
                df[col] = le.inverse_transform(df[col])
        return df