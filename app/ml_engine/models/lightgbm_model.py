import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from app.ml_engine.base import BaseModel
from app.ml_engine.preprocessor import AutoPreprocessor


class LightGBMWrapper(BaseModel):
     def __init__(self, task_type='classification'):
          """
          LightGBM modelini başlatır.
          """
          super().__init__()
          self.task_type = task_type
          param_distributions = {
               'n_estimators': [100, 200, 500],      # Ağaç sayısı
               'learning_rate': [0.01, 0.05, 0.1],  
               'num_leaves': [31, 50, 70],           # Yaprak sayısı 
               'max_depth': [-1, 10, 20],            # Ağaç derinliği 
               'subsample': [0.6, 0.8, 1.0],         # Her ağaçta verinin yüzde kaçı kullanılsın?
               'colsample_bytree': [0.6, 0.8, 1.0]
          }
          if task_type == 'classification':
               self.model = RandomizedSearchCV(
                    lgb.LGBMClassifier(random_state=42),
                    param_distributions,
                    n_iter=10,
                    cv=3,
                    n_jobs=-1,
                    random_state=42,
                    verbose=1
          )
          else:
               self.model = RandomizedSearchCV(
                    lgb.LGBMRegressor(random_state=42),
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