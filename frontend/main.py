import streamlit as st
import requests
import json
import pandas as pd

# API bağlantısının adresi
API_URL = "http://127.0.0.1:8000"

# Sayfa ayarları
st.set_page_config(page_title="AutoML Kontrol Paneli", layout="wide", page_icon="🤖")
st.title("AutoML Kontrol Paneli")

# --- SESSION STATE (Hafıza) Başlatma ---
# Sayfalar arası veri taşımak için session state kullanılıyor.
if 'filename' not in st.session_state: st.session_state['filename'] = None # Dosya adı hafızada kaydediliyor.
if 'columns' not in st.session_state: st.session_state['columns'] = [] # Sütun isimleri hafızada kaydediliyor.
if 'model_trained' not in st.session_state: st.session_state['model_trained'] = False # Model eğitildiğinde hafızada kaydediliyor.

# --- YAN MENÜ ---
menu = ["Veri Yükleme ve Analiz", "Model Eğitimi", "Tahminleme"] # Menü seçenekleri
choice = st.sidebar.radio("Adımlar", menu) # Menü seçeneklerinden seçim yapılıyor


# --- SAYFA 1: VERİ YÜKLEME VE ANALİZ (DATA ANALYSIS) ---
if choice == "Veri Yükleme ve Analiz":
    st.header("Veri Yükleme ve Analiz")
    
    uploaded_file = st.file_uploader("Bir CSV dosyası yükleyin", type=["csv"])
    if uploaded_file is not None:
        if st.button("Yükle ve Analiz Et"):
            with st.spinner("Dosya işleniyor..."):
                try:
                    # Dosya yükleme işlemi yapılıyor.
                    files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
                    response = requests.post(f"{API_URL}/upload", files=files)
                    # API'den gelen yanıt kontrolü yapılıyor.
                    if response.status_code == 200:
                        data = response.json()
                        saved_filename = data["saved_filename"]
                        # Dosya adı hafızada kaydediliyor.
                        st.session_state['filename'] = saved_filename
                        st.success(f"Dosya Yüklendi! ID: {saved_filename}")
                        
                        # Analiz etme işlemi yapılıyor.
                        analyze_res = requests.get(f"{API_URL}/analyze/{saved_filename}")
                        # API'den gelen yanıt kontrolü yapılıyor.
                        if analyze_res.status_code == 200:
                            analyze_data = analyze_res.json()                            
                            # Sütun isimleri eğitim sayfasında kullanılmak üzere hafızaya alınıyor.
                            st.session_state['columns'] = analyze_data["summary"]["column_names"]
                            # İstatistikler listeleniyor.
                            summary = analyze_data["summary"]
                            st.write(summary)
                            # Rapor gösteriliyor.
                            st.subheader("Otomatik Analiz Raporu")
                            st.components.v1.iframe(src=analyze_data["report_url"], height=800, scrolling=True)                            
                    else:
                        st.error("Yükleme başarısız.")
                        
                except Exception as e:
                    st.error(f"Hata: {e}")

# --- SAYFA 2: MODEL EĞİTİMİ (TRAINING) ---
elif choice == "Model Eğitimi":
    st.header("Model Arenası (AutoML V2)")
    
    if st.session_state['filename'] is None:
        st.warning("Lütfen önce '1. Veri Yükle & Analiz' menüsünden bir veri seti yükleyin!")
    else:
        st.success(f"Seçili Veri Seti: {st.session_state['filename']}")
        
        # Form Alanı
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("Hedef Sütun (Tahmin Edilecek)", st.session_state['columns'])
        with col2:
            task_type = st.selectbox("Görev Tipi", ["classification", "regression"])
            st.caption("Sınıflandırma (Kategori) | Regresyon (Sayı)")
                    
        st.markdown("---")
        st.info("💡 'Yarışmayı Başlat' dediğinizde; Random Forest, XGBoost ve LightGBM aynı anda eğitilecek ve en iyisi otomatik seçilecektir.")

        if st.button("🏆 Yarışmayı Başlat"):
            with st.spinner("Modeller arenaya çıkıyor... (RF vs XGB vs LGBM) ⏳"):
                
                # Payload artık model_type içermiyor
                payload = {
                    "filename": st.session_state['filename'],
                    "target_column": target_col,
                    "task_type": task_type
                }
                
                try:
                    res = requests.post(f"{API_URL}/train", json=payload)
                    
                    if res.status_code == 200:
                        result = res.json()
                        
                        # --- SONUÇ EKRANI ---
                        st.balloons() # Kutlama efekti 🎉
                        st.success(f"🏁 Yarışma Bitti! Şampiyon: **{result['winner']}**")
                        
                        # 1. Metrik Kartı
                        col_score, col_winner = st.columns(2)
                        with col_score:
                            st.metric(label="En İyi Skor", value=f"{result['best_score']:.4f}")
                        with col_winner:
                            st.metric(label="Kazanan Model", value=result['winner'])
                        
                        # 2. Liderlik Tablosu (Leaderboard)
                        st.subheader("📊 Liderlik Tablosu")
                        
                        # Backend'den gelen listeyi DataFrame'e çevirip tablo yapıyoruz
                        leaderboard_df = pd.DataFrame(result['leaderboard'])
                        
                        # Tabloyu daha şık göstermek için (Skora göre sırala)
                        leaderboard_df = leaderboard_df.sort_values(by="score", ascending=False)
                        st.table(leaderboard_df)
                        
                        # 3. Model Eğitildi Bilgisi
                        st.session_state['model_trained'] = True
                        st.success("En iyi model sisteme kaydedildi. 'Tahminleme' sayfasına geçebilirsiniz.")
                        
                    else:
                        st.error("Eğitim sırasında hata oluştu.")
                        st.write(res.json())
                except Exception as e:
                    st.error(f"Bağlantı hatası: {e}")

# --- SAYFA 3: TAHMİNLEME (PREDICTION) ---
elif choice == "Tahminleme":
    st.header("Tahminleme")
    
    if not st.session_state['model_trained']:
        st.warning("Önce bir model eğitmelisiniz!")
    else:
        st.success(f"Model Hazır: {st.session_state['filename']}")
        
        # Kullanıcının veri girişi için input kutuları oluşturuluyor.
        input_data = {}
        with st.form("prediction_form"):
            st.write("Veri girişi:")
            # Sütunları döngüye alıp input kutuları oluşturuluyor.
            cols = st.columns(3)
            for i, col_name in enumerate(st.session_state['columns']):
                with cols[i % 3]:
                    val = st.text_input(f"{col_name}", key=f"in_{col_name}")
                    if val:
                        # Sayı mı metin mi kontrolü yapılıyor.
                        try:
                            input_data[col_name] = float(val)
                        except:
                            input_data[col_name] = val
            submit_btn = st.form_submit_button("Tahmin Et")
            
            if submit_btn:
                # app/main.py'deki '/predict' endpoint'ine payload gönderiliyor.
                payload = {
                    "filename": st.session_state['filename'],
                    "data": input_data 
                }
                try:
                    res = requests.post(f"{API_URL}/predict", json=payload)
                    # API'den gelen yanıt kontrolü yapılıyor.
                    if res.status_code == 200:
                        result = res.json()
                        st.markdown(f"## Tahmin sonucu: **{result['prediction']}**")
                    else:
                        st.error("Tahmin hatası.") # API'den gelen hata mesajı gösteriliyor.
                        st.write(res.json())
                except Exception as e:
                    st.error(f"Hata: {e}") # Sunucuya bağlantı hatası olduğunda hata mesajı gösteriliyor.