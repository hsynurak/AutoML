import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

class AutoPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pipeline = None
        self.numeric_features = []
        self.categorical_features = []

    def _date_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tarih özellik çıkarımı modülü: Tarih sütunlarını ayırarak yeni featurelar oluşturur.
        """
        df = df.copy()
        
        for col in df.columns:
            # Türü object olan sütunları kontrol et
            if df[col].dtype == 'object':
                try:
                    # Tarihe çevirmeyi dener (ValueError hatası verirse except bloğuna devam eder)
                    df[col] = pd.to_datetime(df[col], errors='raise')
                    # Başarılıysa parçala (Tarih sütunlarını ayırarak yeni featurelar oluşturur.)
                    df[f"{col}_month"] = df[col].dt.month
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_day"] = df[col].dt.day
                    df[f"{col}_day_of_week"] = df[col].dt.dayofweek
                    # Hafta sonu feature'u oluşturur.
                    df[f"{col}_is_weekend"] = np.where(df[col].dt.dayofweek >= 5, 1, 0)
                    # Ay bilgilerini ve get_season_code fonksiyonunu kullanarak mevsim feature'u oluşturur.
                    def get_season_code(month):
                        if month in [12, 1, 2]: return 1
                        elif month in [3, 4, 5]: return 2
                        elif month in [6, 7, 8]: return 3
                        else: return 4
                    df[f"{col}_season"] = df[f"{col}_month"].apply(get_season_code)
                    # Orijinal tarih sütununu siler.
                    df = df.drop(columns=[col])
                # ValueError ve TypeError hatası verirse tarih sütunu olmadığını anlar ve continue ile devam eder.
                except (ValueError, TypeError):
                    continue
        return df


    def fit(self, X, y=None):
        """
        fit metodu ile veri analiz edilir ve dönüştürme kuralları öğrenilir.
        """
        # Oluşturulan _date_engineering fonksiyonu ile tarih sütunları parçalanır ve yeni featurelar oluşturulur.
        X_engineered = self._date_engineering(X.copy())

        # Sayısal ve Kategorik sütunları ayırma işlemi yapılır.
        self.numeric_features = X_engineered.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = X_engineered.select_dtypes(include=['object', 'category']).columns.tolist()

        # Sayısal veriler için pipeline kurulumu yapılır.
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')), # Eksik değerler medyanla doldur (daha güvenli)
            ('scaler', RobustScaler())                     # Outlier sorununa karşı robust ölçekleme yapılır.
        ])

        # Kategorik veriler için pipeline kurulumu yapılır.
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')), # Eksik değerler en sık geçen değerle doldurulur.
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=20)) # En sık geçen 20 kategoriyi alır.
        ])

        # Sayısal ve kategorik verilerin pipeline yapıları birleştirilir.
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )

        # Pipeline eğitilir.
        self.pipeline.fit(X_engineered, y)
        return self


    def transform(self, X):
        """
        Öğrenilen kurallar yeni veriye uygulanır.
        """
        # Oluşturulan _date_engineering fonksiyonu ile tarih sütunları parçalanır ve yeni featurelar oluşturulur.
        X_engineered = self._date_engineering(X.copy())
        # Pipeline dönüşümü yapılır.
        return self.pipeline.transform(X_engineered)

