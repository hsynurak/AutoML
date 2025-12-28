from abc import ABC, abstractmethod
import pandas as pd
import joblib

# ABC (Abstract Base Class): Bu sınıf doğrudan nesne üretmek için değil, yeni modeller oluştururken miras alınması için oluşturulmuştur.
class BaseModel(ABC):
    
    def __init__(self):
        # Model ve Pipeline'ı burada tanımlıyoruz ki tüm çocuk sınıflar bu değişkenlere sahip olsun.
        self.model = None
        self.pipeline = None

    # @abstractmethod: Bu bir emirdir. Bu metodların miras alınan sınıflarda override edilmesi zorunludur. Bu sayede bütün modeller aynı şekilde çalışır.
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass
    @abstractmethod
    def predict(self, X: pd.DataFrame):
        pass

    # Bu metod concrete bir metod olduğu için override edilmesine gerek yok. Bütün modeller aynı şekilde çalışır.
    def save(self, path: str):
        if self.pipeline:
            # Pipeline kaydedilir. Pipeline içinde model ve preprocessor bulunur.
            joblib.dump(self.pipeline, path)
        else:
            raise ValueError("Model henüz eğitilmemiş, kaydedilemez.")