import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from app.ml_engine.base import BaseModel
from app.ml_engine.preprocessor import AutoPreprocessor

class RandomForestWrapper(BaseModel):
    def __init__(self, task_type='classification'):
        super().__init__()
        self.task_type = task_type
        
        # Modeli başlatır ve seçilen görev tipine göre ilgili modelle işlemler yapılır.
        if task_type == 'classification':
            self.model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        else:
            self.model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
            
        self.pipeline = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Modeli eğitir.
        """
        # AutoPreprocessor sınıfından bir örnek oluşturur.
        preprocessor = AutoPreprocessor()
        
        # Pipeline oluşturulur
        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', self.model)])
        
        # Pipeline eğitilir.
        print("Eğitim başlıyor.")
        self.pipeline.fit(X, y)
        print("Eğitim tamamlandı.")

    def predict(self, X: pd.DataFrame):
        """
        Tahmin yapar.
        """
        # Pipeline var mı kontrolü yapılır. Eğer pipeline yoksa hata verilir.
        if self.pipeline is None:
            raise Exception("Model henüz eğitilmedi!")
        return self.pipeline.predict(X)