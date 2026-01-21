import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from app.ml_engine.models.random_forest import RandomForestWrapper
from app.ml_engine.models.lightgbm_model import LightGBMWrapper
from app.ml_engine.models.xgboost_model import XGBoostWrapper

class ModelTrainer:
     """
     Model Yarıştırma ve Yönetim Sınıfı.
     Veriyi böler, birden fazla modeli eğitir, sonuçları kıyaslar ve şampiyonu kaydeder.
     """
     
     def __init__(self, filename: str, target_column: str, task_type: str):
          self.filename = filename
          self.target_column = target_column
          self.task_type = task_type
          self.results = [] # Skorbord
          self.best_model = None
          self.best_score = -float("inf") # Başlangıçta en iyi skor çok düşük olsun

     def _save_winner(self):
          """
          En iyi modeli diske kaydeder.
          """
          if self.best_model is None:
               raise Exception("Hiçbir model başarıyla eğitilemedi!")

          # Klasör yolu
          MODEL_DIR = "data/models"
          os.makedirs(MODEL_DIR, exist_ok=True)
          
          save_path = os.path.join(MODEL_DIR, f"{self.filename.split('.')[0]}_best_model.pkl")
          
          # En iyi modeli kaydet
          joblib.dump(self.best_model.pipeline, save_path)
          
          print(f"ŞAMPİYON: {self.best_model_name} (Skor: {self.best_score:.4f})")
          print(f"Kaydedildi: {save_path}")

          return {
               "status": "success",
               "winner": self.best_model_name,
               "best_score": self.best_score,
               "leaderboard": self.results, # Frontend'de tablo göstermek için
               "model_path": save_path
          }

     def run(self, df: pd.DataFrame):
          """
          Veriyi split etme, model listesini oluşturma, her modeli sırayla eğitme ve test etme, en iyi modeli seçme ve kaydetme işlemlerini başlatır.
          """
          #Veriyi ayır (X, y)
          X = df.drop(columns=[self.target_column])
          y = df[self.target_column]

          if self.task_type == "classification":
               le = LabelEncoder()
               y = le.fit_transform(y)

          # Train / Test split
          X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

          # Model listesi
          models_map = {
               "Random Forest": RandomForestWrapper(task_type=self.task_type),
               "LightGBM": LightGBMWrapper(task_type=self.task_type),
               "XGBoost": XGBoostWrapper(task_type=self.task_type)
          }

          print(f"Yarışma Başlıyor: {self.task_type.upper()} görevi için {len(models_map)} model yarışacak.")

          # Her modeli sırayla eğitme ve test etme
          for name, model_instance in models_map.items():
               try:
                    # Eğitim
                    print(f"Eğitiliyor: {name}...")
                    model_instance.fit(X_train, y_train)

                    # Tahmin
                    y_pred = model_instance.predict(X_test)

                    # Metrik hesaplama
                    score = 0
                    metric_detail = {}

                    if self.task_type == "classification":
                         # Sınıflandırma için: Accuracy (Doğruluk)
                         score = accuracy_score(y_test, y_pred)
                         metric_detail = {"Accuracy": f"%{score*100:.2f}"}
                    else:
                         # Regresyon için: R2 Score (1'e ne kadar yakınsa o kadar iyi)
                         score = r2_score(y_test, y_pred)
                         mae = mean_absolute_error(y_test, y_pred)
                         metric_detail = {"R2 Score": f"{score:.4f}", "MAE": f"{mae:.2f}"}

                    print(f"{name} Tamamlandı. Skor: {score:.4f}")

                    # Skor tablosuna ekle
                    self.results.append({
                         "model": name,
                         "score": score,
                         "metrics": metric_detail
                    })

                    # En iyi modeli seçme
                    if score > self.best_score:
                         self.best_score = score
                         self.best_model = model_instance
                         self.best_model_name = name

               except Exception as e:
                    print(f" {name} patladı: {e}")
                    continue

          # En iyi modeli kaydet
          return self._save_winner()

