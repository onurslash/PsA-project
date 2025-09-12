# PsA‑XAI: Psöriasis→PsA Erken Öngörü 

* Türkiye popülasyonu için **açıklanabilir** PsA risk modeli. EHR tabanlı, LR/ansambl/Transformer modelleri ve **SHAP** ile populasyon + hasta düzeyi açıklamalar.

## 🎯 Amaç ve Kapsam

* **Amaç:** PsO→PsA geçişini erken ve **yorumlanabilir** tahmin etmek.
* **Veri:** Mersin Ü. Hastanesi **EHR**, ≥1000 PsO, ≥8 yıl takip (anonimleştirilmiş).
* **Modeller:** Lojistik regresyon, **RF/XGBoost/LightGBM**, **Transformer**.
* **Açıklanabilirlik:** **SHAP** (populasyon & local).
* **Metrikler:** **AUC ≥ 0.80** hedef; ayrıca **AUPR** ve **F1**.

# PsO → PsA Simülasyonu · XGBoost · SHAP

**Amaç:** Psoriazis (PsO) → Psöriatik Artrit (PsA) riskini **temsili** bir veriyle modelleyip **SHAP** ile hem **küresel** (kohort) hem **bireysel** (tek kayıt) açıklamalar üretmek.  
**Not:** Bu çalışma **simülasyon** amaçlıdır; klinik iddia içermez.

---

## Hızlı Başlangıç

```bash
# 1) Bağımlılıklar
pip install xgboost shap numpy pandas matplotlib

# 2) Çalıştır
python3 simulation.py --output-folder ./cikti

# (İsteğe bağlı parametreler)
python3 simulation.py --output-folder ./cikti --n 500 --seed 42 --dpi 200
```

---

## Komut Satırı Seçenekleri

| Argüman            | Varsayılan | Açıklama                               |
|--------------------|:----------:|----------------------------------------|
| `--output-folder`  |   gerekli  | Çıktı klasörü yolu                     |
| `--n`              |    220     | Örneklem büyüklüğü                     |
| `--seed`           |     7      | Rastgele tohum                         |
| `--dpi`            |    180     | PNG çözünürlük (DPI)                   |

---

## Üretilen Dosyalar

```
cikti/
├── simulasyon_verisi.csv                 # Özellikler + PsA (0/1) etiket
├── 01_shap_beeswarm_TR.png               # Küresel etki (beeswarm)
├── 02_shap_bar_TR.png                    # Ortalama |SHAP| önem sırası (bar)
├── 03_shap_dependence_TOP_TR.png         # En etkili özellik için bağımlılık grafiği
├── 04_shap_waterfall_TR.png              # Bireysel kayıt waterfall (en yüksek riskli örnek)
├── 05_shap_single_barh_TR.png            # Bireysel kayıtta en büyük 10 katkı
└── README.txt                            # Kısa otomatik özet
```

---

## Grafiklerin Kısa Yorumu

- **Beeswarm (01):** Her nokta bir kişiyi temsil eder. X-ekseni **SHAP değeri** (tahmini ↑/↓ ve katkı büyüklüğü), renk **özellik düzeyi**.  
- **Bar (02):** Ortalama **|SHAP|** ile **populasyon önem sırası**.  
- **Dependence (03):** En etkili özellikte değer arttıkça katkı nasıl değişiyor? (olası eşik/kıvrımlar)  
- **Waterfall (04):** **Tek kayıt** için baz değerden tahmine giderken hangi değişken **artırdı/azalttı**?  
- **Top-10 (05):** Aynı kayıtta katkısı en büyük 10 değişken; etiketler “özellik = gözlenen değer”.

---

## İçeride Ne Oluyor?

1. **Simülasyon verisi** üretilir (demografi, BKİ, CRP, hastalık süresi, tırnak tutulumu, vb.).  
2. Doğrusal skor + küçük **gürültü** ile **ikili etiket** (PsA=0/1) tanımlanır (tamamen temsili).  
3. **XGBoost** modeli eğitilir, **TreeSHAP** ile SHAP değerleri hesaplanır.  
4. En yüksek PsA olasılıklı **tek kayıt** seçilip **waterfall** ve **top-10** grafikleri çizilir.

---

## Sorun Giderme

- **Türkçe karakter/encoding** uyarıları alırsanız:
  - Komutu `python` yerine **`python3`** ile çalıştırın.
  - Gerekirse şu şekilde deneyin:
    ```bash
    env PYTHONIOENCODING=UTF-8 python3 simulation.py --output-folder ./cikti
    ```
  - Editörde dosyanın **UTF-8** olarak kaydedildiğini doğrulayın.
  - Alternatif olarak **ASCII-güvenli** (Türkçe karakterleri Unicode escape’lerle yazan) sürümü kullanın.

- **`xgboost` veya `shap` bulunamadı**:
  ```bash
  pip install xgboost shap
  ```

- **Başsız sunucuda matplotlib**:
  - Betik zaten `Agg` backend’i kullanır; yine de sorun olursa `matplotlib`’i güncelleyin.

---

## Yeniden Üretilebilirlik

- Tohum (`--seed`) parametresi ile deney **tekrarlanabilir**.  
- Çıktı klasöründe **ham veri** (`simulasyon_verisi.csv`) saklanır.

---

## Lisans ve Uyarı

- Kod **araştırma/öğretim** amaçlıdır.  
- Simülasyon varsayımsaldır; **klinik karar** için kullanılmaz.

