# M3 Pro (MPS) için SSL hatası riskine karşı 
# bu kodun kalması iyi bir alışkanlıktır.
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from ultralytics import YOLO
import cv2

# --- 1. AYARLAR ---
# Model 1: Sizin özel baret modeliniz
# Bu dosyanın 'app.py' ile AYNI KLASÖRDE olduğundan emin olun
baret_model_path = r'best.pt'

# Model 2: İnsanları tanıyan genel YOLO modeli
# Bu dosyanın da 'app.py' ile AYNI KLASÖRDE olduğundan emin olun
insan_model_path = r'yolov8n.pt' 

video_source = 0 # 0 = Varsayılan Webcam
# --------------------

# --- Yardımcı Fonksiyon: İki kutu üst üste mi? ---
def check_overlap(insan_box, baret_box):
    """
    Basit bir "içinde mi" kontrolü. 
    Baret kutusunun merkezi, insan kutusunun içindeyse True döner.
    """
    # Baret kutusunun merkezi (bx, by)
    bx_center = (baret_box[0] + baret_box[2]) / 2
    by_center = (baret_box[1] + baret_box[3]) / 2

    # İnsan kutusunun sınırları (ix1, iy1, ix2, iy2)
    ix1, iy1, ix2, iy2 = insan_box

    # Baret merkezi, insanın içindeyse
    if ix1 < bx_center < ix2 and iy1 < by_center < iy2:
        return True
    return False
# -----------------------------------------------

print("Modeller yükleniyor...")
# 2. İKİ MODELİ DE AYRI AYRI YÜKLE
# M3 Pro'da bu satırlar 'device=mps' (GPU) olarak çalışacaktır
try:
    model_baret = YOLO(baret_model_path)
    model_insan = YOLO(insan_model_path)
except Exception as e:
    print(f"HATA: Modeller yüklenemedi. 'best.pt' ve 'yolov8n.pt' dosyalarının bu klasörde olduğundan emin olun.")
    print(f"Hata detayı: {e}")
    exit()

print("Video kaynağı açılıyor...")
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print(f"HATA: Video kaynağı ({video_source}) açılamadı.")
    exit()

print("Çift-Modelli Akıllı alarm sistemi başlatıldı. Çıkmak için 'q' tuşuna basın...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. MODEL 1'İ ÇALIŞTIR (BARETLER İÇİN)
    # verbose=False terminalde sürekli çıktı vermesini engeller
    baret_results = model_baret(frame, stream=True, verbose=False)
    
    # 4. MODEL 2'Yİ ÇALIŞTIR (İNSANLAR İÇİN)
    # classes=0: YoloV8'e 80 sınıf içinden SADECE 0 ID'li sınıfı (insan) bulmasını söyler.
    insan_results = model_insan(frame, stream=True, classes=0, verbose=False)

    annotated_frame = frame
    baret_kutulari = []
    insan_kutulari = []

    # 5. İki modelin de sonuçlarını topla
    for r in baret_results:
        for box in r.boxes.cpu().numpy():
            baret_kutulari.append(box.xyxy[0]) # [x1, y1, x2, y2]
            
    for r in insan_results:
        for box in r.boxes.cpu().numpy():
            insan_kutulari.append(box.xyxy[0]) # [x1, y1, x2, y2]

    # 6. ASIL İŞ GÜVENLİĞİ ALARMI MANTIĞI
    toplam_insan = len(insan_kutulari)
    baretsiz_insan_sayisi = 0

    for insan_box in insan_kutulari:
        baretli_mi = False
        for baret_box in baret_kutulari:
            if check_overlap(insan_box, baret_box):
                baretli_mi = True
                break
        
        if not baretli_mi:
            baretsiz_insan_sayisi += 1
            x1, y1, x2, y2 = map(int, insan_box)
            # UYARI: Baretsiz insanı kırmızıya boya
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3) # Kırmızı kutu
            cv2.putText(annotated_frame, 'BARET YOK!', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            # (İsteğe bağlı) Baretli insanı yeşile boya
            x1, y1, x2, y2 = map(int, insan_box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Yeşil kutu

    # Ekrana genel durumu yazdır
    cv2.putText(annotated_frame, f'Toplam Insan: {toplam_insan}', (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    
    if baretsiz_insan_sayisi > 0:
        cv2.putText(annotated_frame, f'BARETSIZ INSAN: {baretsiz_insan_sayisi}', (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3) # Kırmızı
    else:
        cv2.putText(annotated_frame, f'Baretsiz Insan: 0', (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3) # Yeşil

    cv2.imshow("Canli Is Guvenligi - CIFT MODEL (M3 Pro)", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()