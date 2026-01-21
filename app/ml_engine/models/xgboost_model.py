import pandas as pd
import joblib
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from app.ml_engine.base import BaseModel
from app.ml_engine.preprocessor import AutoPreprocessor

class XGBoostWrapper(BaseModel):
     def __init__(self, task_type='classification'):
          """
          XGBoost modelini başlatır.
          """
          super().__init__()
          self.task_type = task_type
          param_distributions = {
               'max_depth': [3, 5, 7, 10],
               'learning_rate': [0.01, 0.05, 0.1],
               'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               'n_estimators': [50, 100, 200, 500]
          }
          if task_type == 'classification':
               self.model = RandomizedSearchCV(
                    xgb.XGBClassifier(random_state=42),
                    param_distributions,
                    n_iter=10, 
                    cv=3,
                    n_jobs=-1,
                    random_state=42,
                    verbose=1
               )
          else:
               self.model = RandomizedSearchCV(
                    xgb.XGBRegressor(random_state=42),
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
          self.pipeline.fit(X, y)

     def predict(self, X: pd.DataFrame):
          """
          Tahmin yapar.
          """
          if self.pipeline is None:
               raise Exception("Model henüz eğitilmedi!")
          return self.pipeline.predict(X)