import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from app.ml_engine.base import BaseModel
from app.ml_engine.preprocessor import AutoPreprocessor

class RandomForestWrapper(BaseModel):
    def __init__(self, task_type='classification'):
        """
        Random Forest modelini başlatır.
        """
        super().__init__()
        self.task_type = task_type
        param_distributions = {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 8]
        }
        if task_type == 'classification':
            self.model = RandomizedSearchCV(
                RandomForestClassifier(random_state=42), 
                param_distributions, 
                n_iter=10,
                cv=3,
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
        else:
            self.model = RandomizedSearchCV(
                RandomForestRegressor(random_state=42), 
                param_distributions, 
                n_iter=10, 
                cv=3, 
                n_jobs=-1, 
                random_state=42, 
                verbose=1
            )
        self.pipeline = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Modeli eğitir. 
        """
        preprocessor = AutoPreprocessor()
        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', self.model)])
        print("Eğitim başlıyor.")
        self.pipeline.fit(X, y)
        print("Eğitim tamamlandı.")

    def predict(self, X: pd.DataFrame):
        """
        Tahmin yapar.
        """
        if self.pipeline is None:
            raise Exception("Model henüz eğitilmedi!")
        return self.pipeline.predict(X)