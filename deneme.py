# --- ÇÖZÜM 1: SSL HATASINI AŞMAK IÇIN ---
# Bu iki satırı en başa ekleyin
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# ----------------------------------------

from ultralytics import YOLO
import cv2

# --- 1. AYARLAR ---
# ÇÖZÜM 2: SyntaxWarning'i düzeltmek için tırnak başına 'r' ekleyin
model_path = r'C:\baret_dataset\runs\detect\train3\weights\best.pt'

# Webcam kullanmak için: 0
# Video dosyası kullanmak için: "videom.mp4"
video_source = 0 
# --------------------


# 2. Modeli yükle
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"HATA: Model yüklenemedi. Yolunuzu kontrol edin: {model_path}")
    print(f"Hata detayı: {e}")
    exit()

# 3. Video kaynağını aç
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print(f"HATA: Video kaynağı ({video_source}) açılamadı.")
    exit()

print("Webcam başlatıldı. Çıkmak için 'q' tuşuna basın...")

# 4. Videoyu kare kare işle
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video akışı bitti veya okunamadı.")
        break

    # 5. Modeli bu kare üzerinde çalıştır (Tahmin)
    results = model(frame, stream=True)

    # 6. Sonuçları al ve çizdir
    annotated_frame = frame
    for r in results:
        annotated_frame = r.plot() # Kutuları ve etiketleri kare üzerine çiz
        
        # 7. İŞ GÜVENLİĞİ MANTIĞI
        baret_sayisi = len(r.boxes)
        
        # Ekrana baret sayısını yazdır
        cv2.putText(annotated_frame, f'Tespit Edilen Baret: {baret_sayisi}', (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3) # Yeşil renk, kalın

        if baret_sayisi == 0:
            # (Burada "insan var mı?" diye de kontrol edip alarm verebilirsiniz)
            # Şimdilik sadece uyarı yazalım
            cv2.putText(annotated_frame, 'DIKKAT: BARET TESPIT EDILMEDI!', (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3) # Kırmızı renk, kalın


    # 8. İşlenmiş kareyi göster
    cv2.imshow("Canli Is Guvenligi - Baret Tespiti", annotated_frame)

    # 9. Çıkış için 'q' tuşuna basılmasını kontrol et
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 10. Temizlik
cap.release()
cv2.destroyAllWindows()