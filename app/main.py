import os
import shutil
import uuid 
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
import uvicorn
from fastapi.staticfiles import StaticFiles 
from app.ml_engine.analyzer import DataAnalyzer 
import pandas as pd
from pydantic import BaseModel as PydanticSchema 
from app.ml_engine.models.random_forest import RandomForestWrapper 
import joblib 
from typing import Dict, Any 
import logging

# --- LOGGING AYARLARI ---
# Belirli önemli işlemleri loglayarak görüntülemek ve kayıt tutmak için kullanılır.
logging.basicConfig(
     level=logging.INFO,
     format="%(asctime)s [%(levelname)s] %(message)s",
     handlers=[
          logging.FileHandler("app_logs.log"), # Logları dosyaya kaydetme işlemini yapar.
          logging.StreamHandler()              # Logları terminale basma işlemini yapar.
     ]
)
logger = logging.getLogger(__name__)


# --- FASTAPI UYGULAMASI ---
# FastAPI uygulamasını başlatır ve API'nin temel bilgilerini ayarlar.
app = FastAPI(
     title="AutoML API",
     description="Otomatik Makine Öğrenmesi ve Tahminleme Servisi",
     version="0.1.0"
)

# --- GLOBAL SABİT DEĞİŞKENLER ---
# Veri yüklemek için klasör yolunu belirler.
BASE_DIR = "data"
UPLOAD_DIR = os.path.join(BASE_DIR, "raw")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
MODEL_DIR = os.path.join(BASE_DIR, "models")
# Başlangıçta sabit değişkenlerde belirtilen klasörün varlığını kontrol ediyor, yoksa oluşturuyor.
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# *** STATIC FILES AYARI ***
     # REPORT_DIR klasörünü '/reports' adresiyle dışarı açıyoruz. (data/reports/rapor.html -> http://localhost:8000/reports/rapor.html)
app.mount("/reports", StaticFiles(directory=REPORT_DIR), name="reports")


# --- ENDPOINTLER ---

# *** Health Check Endpoint'i ***
# Health Check Endpoint'i: Sistemin ayakta olup olmadığını kontrol eder.
@app.get("/")
async def root():
     return {
          "status": "active", 
          "message": "AutoML API sistemine hoş geldiniz! 🚀"
     }

# *** Upload Endpoint'i ***
     # Upload Endpoint'i: Kullanıcıdan CSV dosyasını alır ve 'data/raw' klasörüne kaydeder.
     # Bu endpoint, kullanıcının CSV dosyasını yüklemesini sağlar. Sadece .csv uzantılı dosyalar kabul eder.(Validation)
     # Dosya ismini benzersiz(unique) yapar. Böylece aynı isimli dosyaların birbirine karışmasını engeller.(Collision Prevention)
     # Büyük dosyaları parça parça yazar. Böylece memory efficiency sağlanır.
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
     # 1. Validation: Dosya tipi kontrolü yapılıyor. Dosya tipi .csv dışında ise hata veriliyor.
     if not file.filename.endswith('.csv'):
          raise HTTPException(status_code=400, detail="Sadece .csv uzantılı dosyalar kabul edilir.")

     # 2. Unique Filename: Aynı isimli dosyaların birbirine karışmasını engelliyor. (ÖR: 'veri.csv' -> 'veri_a1b2c3d4.csv')
     file_extension = file.filename.split(".")[-1]
     original_name = file.filename.split(".")[0]
     unique_filename = f"{original_name}_{uuid.uuid4().hex[:8]}.{file_extension}" # uuid: unique id oluşturmak için kullanılır.
     file_path = os.path.join(UPLOAD_DIR, unique_filename)

     # 3. Memory Efficiency: Dosyayı RAM'e tek seferde yüklemek yerine, 'chunk'lar (küçük parçalar) halinde okuyup diske yazar.
          # Bu sayede GB'larca büyüklükteki dosyalar bile RAM'i şişirmeden kaydedilir.
     try:
          with open(file_path, "wb") as buffer:
               shutil.copyfileobj(file.file, buffer) #Dosyayı 'chunk'lar (küçük parçalar) halinde okuyup diske yazar.
     except Exception as e:
          # Olası disk hatası veya yazma hataları için hata veriliyor.
          raise HTTPException(status_code=500, detail=f"Dosya kaydedilirken hata oluştu: {str(e)}")
     finally:
          await file.close()
     logger.info(f"Dosya başarıyla kaydedildi: {unique_filename}") # Dosya bağlantısı kapatıldıktan sonra loglama işlemi yapılıyor.
     return {
          "status": "success",
          "original_filename": file.filename,
          "saved_filename": unique_filename,
          "file_path": file_path,
          "message": "Dosya başarıyla yüklendi ve kuyruğa hazırlandı."
     }




# *** ANALİZ ENDPOINT'İ ***
     # Yüklenmiş bir dosyanın ismini alır, analiz eder ve raporun linkini döner.
@app.get("/analyze/{filename}")
async def analyze_data(filename: str, request: Request):
     # Veriyi okumak için dosya yolu oluşturuluyor.
     file_path = os.path.join(UPLOAD_DIR, filename) 
     # Dosya var mı kontrolü yapılıyor. Eğer dosya yoksa hata veriliyor.
     if not os.path.exists(file_path):
          raise HTTPException(status_code=404, detail="Dosya bulunamadı. Önce yükleme yapın.")
     # Dosya var ise analiz işlemi başlatılıyor.
     try:
          # DataAnalyzer sınıfından bir örnek oluşturuluyor.
          analyzer = DataAnalyzer(file_path)
          # HTML raporu üretiliyor.
          report_name = analyzer.generate_report()
          # JSON özeti alınıyor.
          summary = analyzer.get_summary_json()
          # Base URL'i alınıyor ve report_url oluşturuluyor.
          base_url = str(request.base_url)
          report_url = f"{base_url}reports/{report_name}"
          return {
               "status": "success",
               "summary": summary,
               "report_url": report_url,
               "message": "Analiz tamamlandı."
          }
     except Exception as e:
          logger.error(f"Analiz hatası: {str(e)}") # Analiz hatası olduğunda uyarı veriliyor ve loglama işlemi yapılıyor.
          raise HTTPException(status_code=500, detail=f"Analiz hatası: {str(e)}")

# --- EĞİTİM İÇİN VERİ ŞEMASI ---
     # Kullanıcıdan gelen verilerin türleri hakkında validasyon işlemleri yapılıyor.
class TrainRequest(PydanticSchema):
     filename: str          # Örn: veri_a1b2.csv
     target_column: str     # Örn: Fiyat
     task_type: str         # 'classification' veya 'regression'
     model_type: str = "random_forest" # İlk etap için varsayılan değer

# --- EĞİTİM ENDPOINT'İ ---
     # Belirtilen veri ve hedef sütun ile model eğitimini başlatır. Sonuçta eğitilmiş modeli (.pkl) diske kaydeder.
@app.post("/train")
async def train_model(request: TrainRequest):
     # Veriyi okumak ve modeli kaydetmek için dosya yolunu oluşturuluyor.
     file_path = os.path.join(UPLOAD_DIR, request.filename)
     model_save_path = os.path.join(MODEL_DIR, f"{request.filename.split('.')[0]}_model.pkl")
     # Dosya var mı kontrolü yapılıyor. Eğer dosya yoksa hata veriliyor.
     if not os.path.exists(file_path):
          raise HTTPException(status_code=404, detail="Dosya bulunamadı.") 
     try:
          # Belirtilen dosya yolu ile veri okunuyor.
          df = pd.read_csv(file_path)
          # Hedef sütun kontrolü yapılıyor. Hedef sütun veride bulunamadığında hata veriliyor.
          if request.target_column not in df.columns:
               raise HTTPException(status_code=400, detail=f"Hedef sütun '{request.target_column}' veride bulunamadı.")
          # X ve y Ayrımı yapılıyor. Hedef sütun dışındaki sütunlar X, hedef sütun y değişkenine atanıyor.
          X = df.drop(columns=[request.target_column])
          y = df[request.target_column]
          # Model başlatılıyor. RandomForestWrapper sınıfından "model" adında bir örnek oluşturuluyor.
          if request.model_type == "random_forest":
               model = RandomForestWrapper(task_type=request.task_type)
          else:
               raise HTTPException(status_code=400, detail="Şimdilik sadece 'random_forest' destekleniyor.")

          model.fit(X, y) # Model eğitiliyor.
          model.save(model_save_path) # Model kaydediliyor.
          # Model eğitildiğinde başarılı olduğu bilgisi dönülüyor.
          return {
               "status": "success",
               "model_path": model_save_path,
               "features": list(X.columns),
               "target": request.target_column,
               "message": "Model başarıyla eğitildi ve kaydedildi!"
          }
     except Exception as e:
          # Hata detayını gösteriliyor ve loglama işlemi yapılıyor.
          logger.error(f"Eğitim hatası: {str(e)}")
          raise HTTPException(status_code=500, detail=f"Eğitim sırasında hata: {str(e)}")

# --- TAHMİN İÇİN VERİ ŞEMASI ---
     # Kullanıcıdan gelen verilerin türleri hakkında validasyon işlemleri yapılıyor.
class PredictRequest(PydanticSchema):
     filename: str          # Hangi modeli kullanacağız? (Örn: veri_a1b2.csv)
     data: Dict[str, Any]   # Tek satırlık veri (Örn: {"yas": 25, "maas": 5000})

# --- TAHMİN ENDPOINT'İ ---
     # Kaydedilmiş modeli yükler ve gönderilen veri için tahmin üretir.
@app.post("/predict")
async def predict(request: PredictRequest):
     # Model dosyasının yolu oluşturuluyor.
     base_name = request.filename.split('.')[0]
     model_path = os.path.join(MODEL_DIR, f"{base_name}_model.pkl")
     # Model dosyası var mı kontrolü yapılıyor. Eğer dosya yoksa hata veriliyor.
     if not os.path.exists(model_path):
          raise HTTPException(status_code=404, detail="Bu dosya için eğitilmiş model bulunamadı.")
     # Model dosyası var ise model yükleniyor.
     try:
          # joblib ile dosya yolu belirtilen model yükleniyor.
          full_pipeline = joblib.load(model_path)
          # Gelen JSON'u DataFrame'e çeviriliyor.
          input_df = pd.DataFrame([request.data])
          # Tahmin yapılıyor.
          prediction = full_pipeline.predict(input_df)
          pred_result = prediction[0]
          if hasattr(pred_result, "item"):
               pred_result = pred_result.item()
          return {
               "status": "success",
               "prediction": pred_result,
               "message": "Tahmin başarıyla oluşturuldu!"
          }
     except Exception as e:
          # Hata detayı gösteriliyor ve loglama işlemi yapılıyor.
          logger.error(f"Tahmin hatası: {str(e)}")
          raise HTTPException(status_code=500, detail=f"Tahmin hatası: {str(e)}")

if __name__ == "__main__":
    # Dosya doğrudan çalıştırılırsa sunucuyu ayağa kaldırıyor.
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)