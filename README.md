# gnn-electricity-price-forecasting
**Elektrik Fiyat Tahmini için Graph Neural Networks (PTF) — GAT Tabanlı Yaklaşım**

Bu repo, “Derin Öğrenme ve Uygulamaları” dersi kapsamında hazırlanmış nihai projeyi içerir. Amaç, Türkiye Gün Öncesi Elektrik Piyasası **PTF** (Piyasa Takas Fiyatı) değerini; farklı üretim kaynakları arasındaki etkileşimleri **Graph Neural Networks (GNN)** ve özellikle **Graph Attention Networks (GAT)** ile modelleyerek tahmin etmektir.

> Not: Bu çalışma **regresyon** problemidir. Bu nedenle sınıflandırma metrikleri (accuracy/IoU/Dice/F1 vb.) yerine **MAE, RMSE, R²** raporlanır.

---

## 1) Problem Tanımı
Elektrik fiyatları; hidroelektrik, rüzgar, güneş ve termik santraller gibi çok sayıda üretim kaynağının **birbirleriyle olan karmaşık etkileşimlerinden** etkilenir. Klasik makine öğrenmesi yaklaşımları bu kaynakları çoğunlukla bağımsız özellikler gibi ele alır.

Bu projede problem bir **graf öğrenme** problemi olarak ele alınmıştır:

- **Node (Düğüm):** Elektrik üretim kaynakları (HES, RES, GES, JES, Kömür, Doğalgaz, vb.)
- **Edge (Kenar):** Kaynaklar arası etkileşim/bağımlılık (tam bağlı directed graph)
- **Hedef:** **PTF** fiyatının tahmini (graph-level regression)

---

## 2) Kullanılan Veri Seti ve Ön İşleme Adımları

### 2.1 Veri Seti
- **Kaynak:** TEİAŞ Şeffaflık Platformu (TPYS)
- **Zaman çözünürlüğü:** 6 saatlik
- **Zaman aralığı:** ~1 yıl
- **Hedef değişken:** PTF
- **Girdi değişkenleri (örnek):**
  - Hidroelektrik (HES)
  - Rüzgar (RES)
  - Güneş (GES)
  - Jeotermal (JES)
  - Biyokütle / Biyogaz
  - Doğalgaz
  - Kömür
  - Sıvı Yakıt / LNG (varsa)

> Ham veri dosyası (https://docs.google.com/spreadsheets/d/1dsucGFk5w1AHpbjMlVE3tYEV6pnFGl1b/edit?usp=sharing&ouid=117161300707052523753&rtpof=true&sd=true)

### 2.2 Ön İşleme
1. Tarih + saat kolonlarından **datetime** oluşturma  
2. Zaman sıralamasına göre **sorting**  
3. **Eksik veri** temizleme  
4. Özellik mühendisliği:
   - Saat ve haftanın günü için **sin/cos kodlama**
   - **Gecikmeli fiyat** özelliği: `prev_price`
5. Zaman bazlı train/test bölünmesi (**%80 / %20**)
6. Standardizasyon / normalizasyon:
   - scaler **yalnızca train verisi ile fit** edilir, test’e uygulanır

---

## 3) Graph (Graf) Temsili

### 3.1 Düğüm ve Kenar Tanımı
- **Düğümler:** üretim kaynakları
- **Kenarlar:** tüm düğümler arasında **tam bağlı (directed)** graph  
  (GAT zaten her komşuya eşit bakmadığı için, ağırlıklandırmayı attention öğrenir.)

### 3.2 Node Feature (Düğüm Özellikleri)
Her node için özellik vektörü:
- Son **L = 24** zaman adımına ait üretim değerleri (kısa vadeli geçmiş)
- Zaman özellikleri: **sin/cos saat** ve **sin/cos gün**
- Bir önceki zaman adımının fiyatı: **prev_price**

> Bu yapı sayesinde model, hem **zamansal örüntüyü** hem de **kaynaklar arası bağımlılıkları** birlikte öğrenir.

### 3.3 Edge Feature (Kenar Özellikleri)
Bu projede edge feature’lar sabit bir sayısal vektör olarak verilmez; ilişkiler **GAT attention ağırlıkları** ile dinamik olarak öğrenilir.

---

## 4) Model Mimarisi ve Yaklaşımın Gerekçesi

### 4.1 Neden GNN / GAT?
- Elektrik sistemi doğal olarak **ağ (network)** yapısına sahiptir.
- Kaynaklar arası etkileşimler sabit değildir; **GAT** komşuları dinamik şekilde ağırlıklandırır.
- Attention ağırlıkları ile **yorumlanabilirlik** sağlar.

### 4.2 Mimarî 
- `GATConv (multi-head)`
- `ELU`
- `GATConv (single-head)`
- `Global Mean Pooling`
- `Linear` (regression head) → **PTF**

Eğitim:
- **Kayıp fonksiyonu:** MSE
- **Optimizasyon:** Adam
- **Epoch:** 25 (varsayılan)

---

## 5) Baseline (Karşılaştırma Modeli)
Karşılaştırma amacıyla **Ridge Regression** baseline kullanılmıştır.

Örnek baseline test sonuçları:
- RMSE ≈ 614
- MAE ≈ 482
- R² ≈ -0.04

> GAT tabanlı model, baseline’a kıyasla belirgin performans artışı sağlamayı hedefler.

---

## 6) Eğitim, Doğrulama ve Test Süreci
- Model belirlenen epoch sayısı boyunca eğitilir
- Test seti üzerinde değerlendirme yapılır
- Kullanılan metrikler:
  - **MAE**
  - **RMSE**
  - **R²**

---

## 7) Model Çıktıları (Repo İçeriğiyle Uyumlu)
Tüm çıktılar `results/` klasörü altında tutulacak şekilde tasarlanmıştır:

- **Nicel metrikler:** `results/metrics.json`
- **Eğitim grafikleri:**
  - `results/figures/loss_curve.png`
  - `results/figures/error_metrics_curve.png`
  - `results/figures/r2_curve.png`
- **Test inference görselleri:**
  - `results/figures/true_vs_pred_test.png`
  - (isteğe bağlı) `results/figures/baseline_true_vs_pred.png`
- **Attention görselleri:**
  - `results/figures/attention_res.png`
  - `results/figures/attention_ges.png`

> Eğer şu an repoda *******************************************************************************************************************

---

## 8) Attention Analizi
İlk GAT katmanındaki attention ağırlıkları analiz edilerek:
- Her üretim kaynağının diğer kaynaklara etkisi (outgoing)
- Diğer kaynaklardan aldığı etkiler (incoming)
- Yenilenebilir kaynakların (RES, GES) fiyat üzerindeki rolü
yorumlanır.

---

## 9) Çalıştırma Talimatları

### Google Colab 
1. Notebook dosyasını Colab’da açın
2. Veri seti Excel dosyasını drive dan yükleyin
3. Hücreleri sırasıyla çalıştırın  
4. Çıktılar `results/` altına kaydedilir


