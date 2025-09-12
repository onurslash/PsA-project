# PsA‑XAI: Psöriasis→PsA Erken Öngörü 

* Türkiye popülasyonu için **açıklanabilir** PsA risk modeli. EHR tabanlı, LR/ansambl/Transformer modelleri ve **SHAP** ile populasyon + hasta düzeyi açıklamalar.

## 🎯 Amaç ve Kapsam

* **Amaç:** PsO→PsA geçişini erken ve **yorumlanabilir** tahmin etmek.
* **Veri:** Mersin Ü. Hastanesi **EHR**, ≥1000 PsO, ≥8 yıl takip (anonimleştirilmiş).
* **Modeller:** Lojistik regresyon, **RF/XGBoost/LightGBM**, **Transformer**.
* **Açıklanabilirlik:** **SHAP** (global & local).
* **Metrikler:** **AUC ≥ 0.80** hedef; ayrıca **AUPR** ve **F1**.
