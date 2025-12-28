import os
import pandas as pd
import io

class DataAnalyzer:
    """
    Pandas tabanlı, hata vermeyen, hafif veri analiz sınıfıdır.
    """
    
    def __init__(self, data_path: str):
        # Dosya yolunu kontrol eder. Eğer dosya yoksa hata verir.
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {data_path}")
        # Dosya yolu hafızaya alınır.
        self.data_path = data_path
        try:
            self.df = pd.read_csv(data_path)
        except Exception as e:
            raise Exception(f"CSV dosyası okunamadı: {e}")

    def generate_report(self, output_dir="data/reports"):
        """
        Basit HTML raporu oluşturur. Veri analizi yapılır ve rapor oluşturulur.
        """
        # Klasör yoksa oluştur
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.basename(self.data_path).split(".")[0]
        report_filename = f"{base_name}_report.html"
        report_path = os.path.join(output_dir, report_filename)
        
        # Eğer rapor zaten varsa, işlem yapmadan ismini döndürür.
        if os.path.exists(report_path):
             return report_filename

        # Tabloları HTML'e çevirir.
        desc_html = self.df.describe().to_html(classes="table table-striped table-bordered", border=0)
        head_html = self.df.head().to_html(classes="table table-hover", border=0)
        dtypes_html = self.df.dtypes.to_frame(name="Veri Tipi").to_html(classes="table", border=0)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Analiz: {base_name}</title>
            <meta charset="utf-8">
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ background-color: #f4f6f9; padding: 20px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
                .container {{ max-width: 1100px; margin: auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); }}
                h1 {{ color: #2c3e50; text-align: center; margin-bottom: 40px; font-weight: 700; }}
                h3 {{ color: #34495e; border-left: 5px solid #3498db; padding-left: 15px; margin-top: 40px; margin-bottom: 20px; }}
                .metric-card {{ background: #ffffff; border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px; text-align: center; transition: transform 0.2s; }}
                .metric-card:hover {{ transform: translateY(-5px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
                .metric-val {{ font-size: 28px; font-weight: bold; color: #3498db; margin-top: 10px; }}
                .metric-label {{ color: #7f8c8d; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }}
                .table-responsive {{ margin-top: 15px; border-radius: 8px; overflow: hidden; }}
                thead {{ background-color: #34495e; color: white; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Veri Analiz Raporu</h1>
                
                <div class="row g-4 mb-5">
                    <div class="col-md-3">
                        <div class="metric-card">
                            <div class="metric-label">Satır Sayısı</div>
                            <div class="metric-val">{self.df.shape[0]}</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <div class="metric-label">Sütun Sayısı</div>
                            <div class="metric-val">{self.df.shape[1]}</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <div class="metric-label">Eksik Değerler</div>
                            <div class="metric-val">{self.df.isnull().sum().sum()}</div>
                        </div>
                    </div>
                     <div class="col-md-3">
                        <div class="metric-card">
                            <div class="metric-label">Tekrar Eden</div>
                            <div class="metric-val">{self.df.duplicated().sum()}</div>
                        </div>
                    </div>
                </div>

                <h3>👀 İlk 5 Satır (Önizleme)</h3>
                <div class="table-responsive">
                    {head_html}
                </div>

                <h3>🧮 İstatistiksel Özet</h3>
                <div class="table-responsive">
                    {desc_html}
                </div>
                
                <h3>🧬 Sütun Tipleri</h3>
                <div class="table-responsive">
                    {dtypes_html}
                </div>
            </div>
        </body>
        </html>
        """

        # HTML'i kaydet
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return report_filename
    
    def get_summary_json(self):
        """
        Frontend'e (Streamlit) gönderilecek özet bilgi.
        """
        # int64 gibi numpy tiplerini normal int'e çeviriyoruz (JSON hatası almamak için)
        return {
            "rows": int(self.df.shape[0]),
            "columns": int(self.df.shape[1]),
            "column_names": list(self.df.columns),
            "missing_values": int(self.df.isnull().sum().sum()),
            "duplicate_rows": int(self.df.duplicated().sum())
        }