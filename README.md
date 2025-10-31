# İş Güvenliği Baret Tespit Sistemi (YOLOv8)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue.svg?logo=opencv&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-purple.svg)

Gerçek zamanlı bir video akışını analiz ederek baret takmayan personeli tespit eden ve görsel uyarı üreten bir bilgisayarlı görü (Computer Vision) uygulamasıdır.

## 🎯 Projenin Amacı

Bu proje, inşaat sahaları, fabrikalar ve depolar gibi iş güvenliğinin kritik olduğu alanlarda güvenlik protokollerini otomatikleştirmeyi amaçlar. Baret takma zorunluluğunu manuel denetim yerine yapay zeka ile sürekli olarak izleyen bu akıllı sistem, kaza riskini ve potansiyel yaralanmaları proaktif olarak azaltmaya yardımcı olur.

## 🛠️ Teknik Mimari: Çift Modelli Yaklaşım

Sistem, modülerlik ve yeniden etiketleme maliyetini ortadan kaldırmak için iki ayrı YOLOv8 modelini eş zamanlı olarak kullanır. Bu mimari, Apple M Serisi (M3 Pro) çiplerin sağladığı MPS (Metal Performance Shaders) donanım hızlandırması sayesinde yüksek performansla çalışır.

1.  **Baret Modeli (Özel Model):**
    * **Dosya:** `best.pt`
    * **Görev:** Kullanıcının kendi (CVAT ile etiketlenmiş) verisiyle eğitilmiş, **sadece 'baret'** nesnesini yüksek doğrulukla tanımaya odaklanmış özel bir modeldir.

2.  **İnsan Modeli (Genel Model):**
    * **Dosya:** `yolov8n.pt`
    * **Görev:** COCO veri seti üzerinde eğitilmiş, 80 farklı sınıfı tanıyabilen standart YOLO modelidir. Bu model, `classes=0` filtresiyle **sadece 'insan'** tespit etmek için kullanılır.

### İş Akışı

Her bir video karesi (frame), eş zamanlı olarak bu iki sinir ağı modelinden de geçirilir.

1.  `model_insan` tüm insanları tespit eder.
2.  `model_baret` tüm baretleri tespit eder.
3.  Uygulama, tespit edilen her bir `insan` kutusunun koordinatlarını, `baret` kutularının koordinatlarıyla karşılaştırır.
4.  Bir `insan` kutusu ile çakışan (veya içinde bulunan) bir `baret` kutusu yoksa, o personel "BARET YOK!" olarak işaretlenir ve görsel olarak uyarılır (kırmızı kutu).

## ✨ Temel Özellikler

* **Gerçek Zamanlı Tespit:** Apple M3 Pro (MPS) üzerinde akıcı ve yüksek FPS ile çalışır.
* **Akıllı Uyarı Sistemi:** Sadece nesneleri listelemez; insanlar ve baretler arasındaki ilişkiyi analiz ederek "baretsiz personel" tespiti yapar.
* **Modülerlik:** Veri setlerini yeniden etiketlemeye gerek kalmadan baret modelini (daha fazla veri ile) veya insan modelini (örn. `yolov8m.pt` ile) bağımsız olarak iyileştirebilme imkanı.
* **Düşük Yanlış Alarm:** `model_insan` sayesinde, sadece bir baretin yerde durması yerine, baretin bir insanla ilişkisi denetlenir.

## 🚀 Kurulum (macOS / Apple Silicon)

Proje, Apple M serisi (M1, M2, M3) çiplere sahip macOS cihazlar için optimize edilmiştir.

### 1. Proje Dosyalarının Hazırlanması

Projenin çalışması için gerekli olan eğitilmiş model dosyalarını temin edin ve ana proje klasörüne kopyalayın.

```bash
/BaretProjesi/
├── app.py              # Ana uygulama kodu
├── best.pt             # Sizin eğittiğiniz özel baret modeli
├── yolov8n.pt            # Standart YOLOv8 (nano) modeli
├── requirements.txt      # Gerekli kütüphaneler
└── README.md             # Bu dosya
```

### 2. Sanal Ortamın Kurulması (Tavsiye Edilir)

Terminali açın ve proje klasörüne gidin.

```bash
# Proje klasörüne gidin
cd /path/to/BaretProjesi

# 'venv' adında bir sanal ortam oluşturun
python3 -m venv venv

# Sanal ortamı aktive edin
source venv/bin/activate
```

### 3. Bağımlılıkların Yüklenmesi

`requirements.txt` dosyasını kullanarak gerekli tüm Python kütüphanelerini yükleyin. Bu komut, Apple Silicon (MPS) desteğini içeren PyTorch versiyonunu otomatik olarak kuracaktır.

```bash
pip install -r requirements.txt
```

## ⚡️ Uygulamayı Çalıştırma

Tüm bağımlılıklar yüklendikten sonra, uygulamayı aşağıdaki komutla başlatabilirsiniz:

```bash
python3 app.py
```

Uygulama otomatik olarak `device=mps` (M3 Pro GPU) ayarını seçecek ve webcam'inizi (varsayılan `video_source = 0`) açacaktır.

* Çıkmak için, OpenCV tarafından açılan video penceresi odaktayken klavyeden **'q'** tuşuna basın.

## ⚙️ Yapılandırma

Temel ayarlar `app.py` dosyasının en üst kısmındaki `AYARLAR` bölümünden değiştirilebilir:

* `baret_model_path`: Özel baret modelinizin yolu.
* `insan_model_path`: Genel insan modelinizin yolu.
* `video_source`: Video kaynağı.
    * `0`: Varsayılan webcam
    * `"video_dosyasi.mp4"`: Bir video dosyasını işlemek için.
