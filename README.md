# 🏦 FINRISK PRO — Kredi Risk Skor Kartı & Model Validasyon Motoru

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/Model-LightGBM%20%26%20LogReg-success)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![GINI](https://img.shields.io/badge/Test%20GINI-0.739-2563EB)
![PSI](https://img.shields.io/badge/PSI-0.0005%20(Stable)-10B981)

**FINRISK PRO**, bankacılık ve finans sektöründeki kredi tahsis, skorlama ve model validasyon süreçlerini (Basel III / IFRS 9 metodolojisi çerçevesinde) uçtan uca simüle eden bir **Kredi Risk Karar Destek Sistemidir**.

150.000 gözlemli *Give Me Some Credit* veri seti üzerinde eğitilen sistem; veri sızıntısını (data leakage) önleyen deterministik önişleme hattı, sıfırdan geliştirilen **WoE / IV** değişken seçimi, **Bayes önsel olasılık kalibrasyonu**, **PDO (Points to Double the Odds)** puan ölçekleme, **SHAP** tabanlı komite karar açıklamaları ve **Finansal Kâr/Zarar (P&L) kesme eşiği (cut-off) optimizasyonunu** tek bir interaktif uygulamada birleştirir.

---

## 🎯 Temel Özellikler ve Metodoloji

### 1. Veri Sızıntısını Önleyen (Leakage-Safe) Önişleme
Tüm temizlik ve uç değer parametreleri **yalnızca eğitim (%80 Train) setinden** öğrenilir (`compute_cleaning_params`) ve `metadata.pkl` aracılığıyla hem test setine hem de canlı Streamlit arayüzüne uygulanır:
- **Yaş Anomalileri:** `<= 18` veya `> 95` olan kayıtlar eğitim seti medyanı ile düzeltilir.
- **Sistemik Büro Hata Kodları:** Gecikme değişkenlerindeki (`30-59`, `60-89`, `90+` gün) `96` ve `98` kodları temizlenerek eğitim medyanı atanır.
- **Uç Değer Kırpma (Outlier Clipping):** `RevolvingUtilizationOfUnsecuredLines` değişkeni eğitim setinin `%99` persentil sınırında kırpılır.
- **Eksik Veri Atama:** `MonthlyIncome` ve `NumberOfDependents` eksiklikleri eğitim medyanları ile doldurulur.

### 2. Sıfırdan WoE / IV Hesaplaması ve Sıfır Ağırlıklı (Zero-Inflated) Bantlama
Harici kütüphane kullanılmadan `pandas` ve `numpy` ile yazılan `calculate_woe_iv` fonksiyonu:
- $\ln(0)$ ve sıfıra bölme hatalarını önlemek için **Laplace düzeltmesi (`laplace = 0.5`)** uygular:
  $$\text{WoE}_i = \ln\left(\frac{\%\text{Sağlam}_i}{\%\text{Batık}_i}\right), \quad \text{IV} = \sum_{i=1}^{k} (\%\text{Sağlam}_i - \%\text{Batık}_i) \times \text{WoE}_i$$
- Gecikme sayıları gibi gözlemlerin %90'ından fazlasının `0` olduğu değişkenlerde `pd.qcut` 3'ten az banda düştüğünde otomatik olarak kesikli aralıklara (`0`, `1`, `2`, `3+`) geçiş yapar.
- Tahmin gücü eşiğinin altında kalan (`IV < 0.02`) değişkenler modelden otomatik olarak elenir.

### 3. Modelleme ve Bayes Olasılık Kalibrasyonu
- **Ana Motor (LightGBM):** Aşırı öğrenmeyi (overfitting) engellemek için sıkı regülarizasyon parametreleriyle (`max_depth=4`, `num_leaves=15`, `min_child_samples=50`, `subsample=0.8`, `learning_rate=0.03`) eğitilmiştir.
- **Referans Model (Lojistik Regresyon):** `SimpleImputer` + `StandardScaler` + `LogisticRegression` hattı benchmark olarak kurulmuştur.
- **Bayes Önsel Düzeltmesi (Prior Calibration):** Dengesiz veri setinde (`%6.68` temerrüt oranı) `class_weight="balanced"` kullanımından kaynaklanan olasılık şişmesi Bayes teoremi ile gerçek kitle temerrüt oranına geri çekilir:
  $$P_{\text{cal}} = \frac{P_{\text{raw}} \cdot r}{P_{\text{raw}} \cdot r + (1 - P_{\text{raw}})(1 - r)}, \quad r = \frac{\text{Base Rate (0.0668)}}{0.50}$$

### 4. PDO Kredi Skor Kartı ve 3 Kademeli Karar Motoru
Kalibre edilmiş temerrüt olasılıkları (PD), endüstri standardı **PDO (Points to Double the Odds)** yöntemiyle `300 – 850` puan aralığına ölçeklenir (`BASE_SCORE = 600`, `BASE_ODDS = 1:19`, `PDO = 50`):
$$\text{Skor} = \text{Offset} + \text{Factor} \times \ln\left(\frac{1 - \text{PD}}{\text{PD}}\right)$$
- **ONAY (`Skor >= 600`):** Düşük risk profili, standart fiyatlandırma ile otomatik onay.
- **TEMİNATLI İNCELEME (`496 <= Skor < 600`):** Gri bölge (Risk Grubu C); ek teminat veya gelir belgesi ile Kredi Tahsis Komitesi incelemesi.
- **RET (`Skor < 496`):** Yüksek temerrüt riski, otomatik ret önerisi.

---

## 📊 Model Validasyon Sonuçları

| Model | Veri Seti | ROC-AUC | GINI | KS İstatistiği | Brier Score | GINI Gap | PSI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | Train | `0.8710` | `0.7420` | `0.5861` | `0.0558` | `+0.0029 [PASS]` | `0.0005 [Stable]` |
| **LightGBM** | **Test** | **`0.8695`** | **`0.7391`** | **`0.5824`** | **`0.0562`** | `+0.0029 [PASS]` | `0.0005 [Stable]` |
| **Lojistik Regresyon** | Train | `0.8527` | `0.7055` | `0.5512` | `0.0598` | `-0.0121 [PASS]` | `0.0003 [Stable]` |
| **Lojistik Regresyon** | **Test** | **`0.8588`** | **`0.7176`** | **`0.5618`** | **`0.0594`** | `-0.0121 [PASS]` | `0.0003 [Stable]` |

---

## 🖥️ Uygulama Sekmeleri (`app.py`)

1. **Canlı Kredi Karar Motoru:** Sol panelden hazır müşteri profilleri (*Düşük Riskli VIP*, *Gri Bölge / İnceleme*, *Yüksek Riskli*) veya manuel parametrelerle anlık PDO skoru, kalibre PD (%), risk grubu, ibre (gauge) grafiği ve **SHAP** tabanlı Kredi Tahsis Komitesi değerlendirme notu üretir.
2. **EDA & WoE / IV Raporu:** Tüm değişkenlerin IV güç sıralamasını, dağılım histogramlarını ve bant bazlı batık oranı eğrilerini interaktif olarak sunar.
3. **Model Validasyonu & Kalibrasyon:** Train vs Test GINI/KS/AUC karşılaştırmalarını, Overfitting/PSI kontrol kartlarını, Kalibrasyon (Reliability) eğrisini ve Brier Score grafiklerini içerir.
4. **Cut-off Optimizasyonu & P&L Simülatörü:** Sağlam krediden `+$1,000` faiz geliri, batık krediden `-$10,000` anapara kaybı varsayımıyla `%2`'den `%50`'ye kadar tüm PD kesme eşiklerini simüle eder; portföy kârını maksimize eden optimum kesme noktasını (`%18 PD` / `496 Skor`) raporlar.

---

## ⚙️ Kurulum ve Çalıştırma

```bash
# 1. Depoyu klonlayın
git clone [https://github.com/admmdmc3/credit-risk-scorecard.git](https://github.com/admmdmc3/credit-risk-scorecard.git)
cd credit-risk-scorecard

# 2. Sanal ortam oluşturun ve bağımlılıkları yükleyin
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS / Linux
pip install -r requirements.txt

# 3. (İsteğe bağlı) Modeli baştan eğitmek ve artefaktları üretmek için
python train_model.py

# 4. Streamlit karar destek arayüzünü başlatın
streamlit run app.py
```

---

## 📁 Proje Dizin Yapısı

```text
credit_risk_scorecard/
├── app.py                      # Streamlit kurumsal karar destek arayüzü
├── train_model.py              # Uçtan uca eğitim, kalibrasyon ve validasyon hattı
├── data/
│   └── cs-training.csv         # Eğitim ve test veri seti
├── models/
│   ├── lgbm_model.pkl          # Eğitilmiş LightGBM modeli
│   ├── lr_pipeline.pkl         # Benchmark Lojistik Regresyon pipeline'ı
│   ├── metadata.pkl            # Leakage-safe temizlik ve PDO parametreleri
│   ├── iv_table.csv            # Değişken IV değerleri ve güç sınıfları
│   ├── metrics.json            # Train/Test AUC, GINI, KS, Brier ve PSI metrikleri
│   ├── validation_curves.json  # Kalibrasyon ve Cut-off P&L simülasyon verileri
│   └── shap_background.csv     # SHAP TreeExplainer arka plan örneklemi
└── requirements.txt            # Python kütüphane bağımlılıkları
```
