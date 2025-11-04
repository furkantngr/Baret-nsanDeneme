import cv2
from ultralytics import YOLO
import argparse 
import time

# --- AYARLAR ---
MODEL_HELMET_PATH = 'best.pt'     # Sizin özel modeliniz
MODEL_PERSON_PATH = 'yolov8n.pt'  # Standart YOLOv8 Nano

# Güven eşikleri
HELMET_GUVEN_ESIGI = 0.80 # Sadece %80 üzeri baretleri dikkate al
PERSON_GUVEN_ESIGI = 0.20 # İnsan tespiti için minimum eşik (takip için)

# Uyarı Süresi
UYARI_SURESI = 10  # saniye

# Renkler (BGR formatında)
RENKLER = {
    'takan': (0, 255, 0),     # Yeşil (Baret Var)
    'takmayan': (0, 0, 255),   # Kırmızı (Baret Yok)
    'unassigned': (128, 128, 128) # Gri (İlişkisiz nesne)
}

# --- YARDIMCI FONKSİYONLAR ---

def get_bbox_center(bbox):
    """Bir sınırlayıcı kutunun (x1, y1, x2, y2) merkez noktasını (cx, cy) döndürür."""
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy

def is_on_shoulders(head_or_helmet_bbox, person_bbox, top_percentage=0.3):
    """
    Bir kafa/baret kutusunun, bir kişi kutusunun omuzlarında (üst %'lik diliminde)
    olup olmadığını kontrol eder.
    """
    h_cx, h_cy = get_bbox_center(head_or_helmet_bbox)
    p_x1, p_y1, p_x2, p_y2 = person_bbox
    
    # 1. Yatay Kontrol: Kafa/Baret, kişinin yatay sınırları içinde mi?
    is_horizontally_aligned = (p_x1 < h_cx < p_x2)
    
    # 2. Dikey Kontrol: Kafa/Baret, kişinin üst %'lik diliminde mi?
    #    (p_y1 < h_cy < omuz_seviyesi)
    person_height = p_y2 - p_y1
    shoulder_level = p_y1 + (person_height * top_percentage)
    is_vertically_aligned = (p_y1 < h_cy < shoulder_level)
    
    return is_horizontally_aligned and is_vertically_aligned

# --- / YARDIMCI FONKSİYONLAR ---


# --- Argüman Ayrıştırıcı ---
parser = argparse.ArgumentParser(description="YOLOv8 ile Çift Modelli İş Güvenliği Takibi")
parser.add_argument('--source', type=str, default='0',
                    help="Giriş kaynağı: '0' (webcam) veya video yolu ('video.mp4')")
args = parser.parse_args()

# --- Modelleri Yükle ---
try:
    print(f"Kişi modeli ({MODEL_PERSON_PATH}) yükleniyor...")
    model_person = YOLO(MODEL_PERSON_PATH)
    print("Kişi modeli yüklendi.")
    
    print(f"Özel baret modeli ({MODEL_HELMET_PATH}) yükleniyor...")
    model_helmet = YOLO(MODEL_HELMET_PATH)
    print("Özel baret modeli yüklendi.")
except Exception as e:
    print(f"Hata: Modeller yüklenemedi. Dosya yolları doğru mu?")
    print(f"Detay: {e}")
    exit()

# Özel modeldeki 'helmet' sınıfının ID'sini bul
helmet_names = model_helmet.names
HELMET_CLASS_ID = None
for class_id, name in helmet_names.items():
    if name == 'helmet':
        HELMET_CLASS_ID = class_id
        break

if HELMET_CLASS_ID is None:
    print(f"Hata: Özel modelinizde ('{MODEL_HELMET_PATH}') 'helmet' adında bir sınıf bulunamadı.")
    print(f"Bulunan sınıflar: {helmet_names}")
    exit()
else:
    print(f"'helmet' sınıfı ID {HELMET_CLASS_ID} olarak bulundu.")

# COCO modelinde 'person' sınıfı her zaman 0'dır
PERSON_CLASS_ID = 0

# --- Video Kaynağını Başlat ---
source = args.source
is_webcam = source.isdigit()

if is_webcam:
    cap = cv2.VideoCapture(int(source))
    print(f"Kamera {source} başlatılıyor...")
else:
    cap = cv2.VideoCapture(source)
    print(f"Video dosyası '{source}' işleniyor...")

if not cap.isOpened():
    print(f"Hata: Kaynak '{source}' açılamadı.")
    exit()

# --- NESNE TAKİBİ DEĞİŞKENLERİ ---
# Key: person_id
ihlal_takip_listesi = {} 

print("-" * 30)
print(f"[BİLGİ] {UYARI_SURESI} saniye baret takmayan 'KİŞİ ID'leri' için uyarı verilecek.")
print("-" * 30)

while True:
    success, frame = cap.read()
    if not success:
        print("Akış sonlandı.")
        break

    # --- 1. Adım: Her İki Model ile Takip Yap ---
    
    # Kişileri Takip Et (Sadece 'person' sınıfı [0])
    results_person = model_person.track(frame, persist=True, classes=[PERSON_CLASS_ID], 
                                          conf=PERSON_GUVEN_ESIGI, verbose=False)
    
    # Baretleri Takip Et (Sadece 'helmet' sınıfı [HELMET_CLASS_ID])
    results_helmet = model_helmet.track(frame, persist=True, classes=[HELMET_CLASS_ID], 
                                          conf=HELMET_GUVEN_ESIGI, verbose=False)

    # --- 2. Adım: Tespitleri Listelere Aktar ---
    persons = []
    helmets = []
    
    if results_person[0].boxes:
        for box in results_person[0].boxes:
            if box.id is not None:
                persons.append({
                    'id': int(box.id[0]),
                    'bbox': list(map(int, box.xyxy[0])),
                    'conf': float(box.conf[0])
                })

    if results_helmet[0].boxes:
        for box in results_helmet[0].boxes:
            if box.id is not None:
                helmets.append({
                    'id': int(box.id[0]), # Bu ID, helmet modeline aittir
                    'bbox': list(map(int, box.xyxy[0])),
                    'conf': float(box.conf[0])
                })

    # --- 3. Adım: Kişi Merkezli Mantığı Çalıştır ---
    
    baret_takan_sayisi = 0
    baret_takmayan_sayisi = 0
    current_frame_person_ids_no_helmet = set()
    drawn_helmet_ids = set() # Çizilen (ilişkilendirilen) baret ID'leri

    for person in persons:
        person_id = person['id']
        person_bbox = person['bbox']
        
        status = "takmayan" # Güvenlik için varsayılan: Takmıyor
        matched_helmet_bbox = None

        # Bu kişiye uyan bir baret ara
        for helmet in helmets:
            if is_on_shoulders(helmet['bbox'], person_bbox):
                status = "takan"
                matched_helmet_bbox = helmet['bbox']
                drawn_helmet_ids.add(helmet['id']) # Bu bareti 'ilişkili' olarak işaretle
                break # Bu kişi için baret bulundu, aramayı durdur
        
        # Durumu işle
        if status == "takan":
            baret_takan_sayisi += 1
            # Kişinin giydiği bareti yeşil çiz
            label = f"ID {person_id}: BARET TAKIYOR"
            cv2.rectangle(frame, (matched_helmet_bbox[0], matched_helmet_bbox[1]), (matched_helmet_bbox[2], matched_helmet_bbox[3]), RENKLER['takan'], 2)
            cv2.putText(frame, label, (matched_helmet_bbox[0], matched_helmet_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RENKLER['takan'], 2)
        
        else: # status == "takmayan"
            baret_takmayan_sayisi += 1
            current_frame_person_ids_no_helmet.add(person_id)
            
            # Kişinin kendisini kırmızı çiz
            label = f"ID {person_id}: BARET YOK"
            cv2.rectangle(frame, (person_bbox[0], person_bbox[1]), (person_bbox[2], person_bbox[3]), RENKLER['takmayan'], 2)
            cv2.putText(frame, label, (person_bbox[0], person_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RENKLER['takmayan'], 2)

            # --- KİŞİ ID'Sİ BAZLI ZAMANLAYICI ---
            if person_id not in ihlal_takip_listesi:
                ihlal_takip_listesi[person_id] = {'start_time': time.time(), 'warned': False}
            else:
                data = ihlal_takip_listesi[person_id]
                gecen_sure = time.time() - data['start_time']
                
                if gecen_sure > UYARI_SURESI and not data['warned']:
                    print(f"\n[UYARI] {time.strftime('%H:%M:%S')} - KISI ID {person_id} {UYARI_SURESI} saniyedir baret takmiyor!\n")
                    data['warned'] = True

    # --- 4. Adım: İhlal Listesi Temizliği ---
    for person_id in list(ihlal_takip_listesi.keys()):
        if person_id not in current_frame_person_ids_no_helmet:
            if ihlal_takip_listesi[person_id]['warned']:
                print(f"[BİLGİ] KISI ID {person_id} icin ihlal durumu sona erdi.")
            del ihlal_takip_listesi[person_id]

    # --- 5. Adım: İlişkisiz Kalan Baretleri Çiz ---
    for helmet in helmets:
        if helmet['id'] not in drawn_helmet_ids:
            bbox = helmet['bbox']
            label = f"ID {helmet['id']}: Iliskisiz Baret"
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), RENKLER['unassigned'], 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RENKLER['unassigned'], 2)

    # --- 6. Adım: Ekran Bilgileri ---
    text_no_helmet = f"Baret Takmayan Sayisi: {baret_takmayan_sayisi}"
    cv2.putText(frame, text_no_helmet, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, RENKLER['takmayan'], 2, cv2.LINE_AA)
    
    text_helmet = f"Baret Takan Sayisi: {baret_takan_sayisi}"
    cv2.putText(frame, text_helmet, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1, RENKLER['takan'], 2, cv2.LINE_AA)

    cv2.putText(frame, "Cikis: 'q'", (frame.shape[1] - 100, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Is Guvenligi Tespiti (Çift Model)", frame)

    key = cv2.waitKey(1) & 0xFF 
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program sonlandırıldı.")