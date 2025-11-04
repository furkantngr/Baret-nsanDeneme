import sys
import cv2
from ultralytics import YOLO
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QWidget, QPushButton, QHBoxLayout, QTextEdit, 
                             QFileDialog, QProgressBar)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QObject,pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
import logging
# --- AYARLAR ---
# Modellerin yolları
MODEL_HELMET_PATH = 'best.pt'
MODEL_PERSON_PATH = 'yolov8n.pt'

# Güven eşikleri
HELMET_GUVEN_ESIGI = 0.80 
PERSON_GUVEN_ESIGI = 0.25 

# Uyarı Süresi
UYARI_SURESI = 10  # saniye

logging.basicConfig(
    level=logging.INFO,  # Hangi seviyedeki logları tutacağı (INFO ve üzeri)
    format="%(asctime)s [%(levelname)s] - %(message)s", # Log formatı (Tarih - Seviye - Mesaj)
    handlers=[
        logging.FileHandler("is_guvenligi.log", mode='a', encoding='utf-8') # Dosyaya yaz
        # İsteğe bağlı: Terminale de basmak için bunu ekleyebilirsiniz
        # logging.StreamHandler(sys.stdout) 
    ]
)

# Renkler (BGR formatında)
RENKLER = {
    'takan': (0, 255, 0),     # Yeşil
    'takmayan': (0, 0, 255),   # Kırmızı
    'unassigned': (128, 128, 128) # Gri
}

# --- YARDIMCI FONKSİYONLAR ---
# Bu fonksiyonlar değişmeden kalıyor
def get_bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy

def is_on_shoulders(head_or_helmet_bbox, person_bbox, top_percentage=0.3):
    h_cx, h_cy = get_bbox_center(head_or_helmet_bbox)
    p_x1, p_y1, p_x2, p_y2 = person_bbox
    is_horizontally_aligned = (p_x1 < h_cx < p_x2)
    person_height = p_y2 - p_y1
    shoulder_level = p_y1 + (person_height * top_percentage)
    is_vertically_aligned = (p_y1 < h_cy < shoulder_level)
    return is_horizontally_aligned and is_vertically_aligned
# --- / YARDIMCI FONKSİYONLAR ---


# --- VİDEO İŞ PARÇACIĞI (WORKER THREAD) ---
# Tüm ağır video işleme yükü bu sınıfta
class VideoThread(QThread):
    # Arayüzü güncellemek için Sinyaller
    change_pixmap_signal = pyqtSignal(QImage)
    update_counts_signal = pyqtSignal(int, int) # (takan, takmayan)
    alert_signal = pyqtSignal(str) # Terminal uyarısı için
    finished_signal = pyqtSignal() # İşlem bittiğinde

    def __init__(self, source):
        super().__init__()
        self.source = source
        self._running = True
        self.ihlal_takip_listesi = {}
        self.HELMET_CLASS_ID = None

    def load_models(self):
        try:
            self.alert_signal.emit(f"Kişi modeli ({MODEL_PERSON_PATH}) yükleniyor...")
            self.model_person = YOLO(MODEL_PERSON_PATH)
            
            self.alert_signal.emit(f"Özel baret modeli ({MODEL_HELMET_PATH}) yükleniyor...")
            self.model_helmet = YOLO(MODEL_HELMET_PATH)
            
            helmet_names = self.model_helmet.names
            for class_id, name in helmet_names.items():
                if name == 'helmet':
                    self.HELMET_CLASS_ID = class_id
                    break
            
            if self.HELMET_CLASS_ID is None:
                self.alert_signal.emit(f"[HATA] Özel modelde 'helmet' sınıfı bulunamadı. Bulunanlar: {helmet_names}")
                return False
            
            self.alert_signal.emit(f"'helmet' ID {self.HELMET_CLASS_ID} olarak bulundu. Kişi ID 0.")
            return True
        except Exception as e:
            self.alert_signal.emit(f"[HATA] Modeller yüklenemedi: {e}")
            return False

    def run(self):
        if not self.load_models():
            self.finished_signal.emit()
            return # Modeller yüklenemezse thread'i durdur

        # Kaynağı aç
        if self.source.isdigit():
            cap = cv2.VideoCapture(int(self.source))
        else:
            cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            self.alert_signal.emit(f"[HATA] Kaynak açılamadı: {self.source}")
            self.finished_signal.emit()
            return
            
        self.alert_signal.emit(f"İşlem başlatıldı: {self.source}")

        while self._running:
            success, frame = cap.read()
            if not success:
                self.alert_signal.emit("Video akışı sonlandı.")
                break # Video bitti veya kamera kapandı
            
            # --- Burası önceki kodumuzdaki 'while' döngüsü ile aynı ---
            
            # 1. Modelleri çalıştır
            results_person = self.model_person.track(frame, persist=True, classes=[0], 
                                                     conf=PERSON_GUVEN_ESIGI, verbose=False)
            results_helmet = self.model_helmet.track(frame, persist=True, classes=[self.HELMET_CLASS_ID], 
                                                     conf=HELMET_GUVEN_ESIGI, verbose=False)
            
            # 2. Tespitleri listele
            persons = []
            helmets = []
            if results_person[0].boxes:
                for box in results_person[0].boxes:
                    if box.id is not None:
                        persons.append({'id': int(box.id[0]), 'bbox': list(map(int, box.xyxy[0]))})

            if results_helmet[0].boxes:
                for box in results_helmet[0].boxes:
                    if box.id is not None:
                        helmets.append({'id': int(box.id[0]), 'bbox': list(map(int, box.xyxy[0]))})
            
            # 3. Kişi merkezli mantık
            baret_takan_sayisi = 0
            baret_takmayan_sayisi = 0
            current_frame_person_ids_no_helmet = set()
            drawn_helmet_ids = set()

            for person in persons:
                person_id = person['id']
                person_bbox = person['bbox']
                status = "takmayan"
                matched_helmet_bbox = None

                for helmet in helmets:
                    if is_on_shoulders(helmet['bbox'], person_bbox):
                        status = "takan"
                        matched_helmet_bbox = helmet['bbox']
                        drawn_helmet_ids.add(helmet['id'])
                        break
                
                if status == "takan":
                    baret_takan_sayisi += 1
                    label = f"ID {person_id}: BARET TAKIYOR"
                    cv2.rectangle(frame, (matched_helmet_bbox[0], matched_helmet_bbox[1]), (matched_helmet_bbox[2], matched_helmet_bbox[3]), RENKLER['takan'], 2)
                    cv2.putText(frame, label, (matched_helmet_bbox[0], matched_helmet_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RENKLER['takan'], 2)
                else:
                    baret_takmayan_sayisi += 1
                    current_frame_person_ids_no_helmet.add(person_id)
                    label = f"ID {person_id}: BARET YOK"
                    cv2.rectangle(frame, (person_bbox[0], person_bbox[1]), (person_bbox[2], person_bbox[3]), RENKLER['takmayan'], 2)
                    cv2.putText(frame, label, (person_bbox[0], person_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RENKLER['takmayan'], 2)

                    if person_id not in self.ihlal_takip_listesi:
                        self.ihlal_takip_listesi[person_id] = {'start_time': time.time(), 'warned': False}
                    else:
                        data = self.ihlal_takip_listesi[person_id]
                        gecen_sure = time.time() - data['start_time']
                        
                        if gecen_sure > UYARI_SURESI and not data['warned']:
                            # Terminale/Log kutusuna UYARI SİNYALİ gönder
                            self.alert_signal.emit(f"[UYARI] {time.strftime('%H:%M:%S')} - KISI ID {person_id} {UYARI_SURESI} saniyedir baret takmiyor!")
                            data['warned'] = True
            
            # 4. İhlal listesi temizliği
            for person_id in list(self.ihlal_takip_listesi.keys()):
                if person_id not in current_frame_person_ids_no_helmet:
                    if self.ihlal_takip_listesi[person_id]['warned']:
                        self.alert_signal.emit(f"[BİLGİ] KISI ID {person_id} icin ihlal durumu sona erdi.")
                    del self.ihlal_takip_listesi[person_id]
            
            # 5. İlişkisiz baretleri çiz
            for helmet in helmets:
                if helmet['id'] not in drawn_helmet_ids:
                    bbox = helmet['bbox']
                    label = f"ID {helmet['id']}: Iliskisiz Baret"
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), RENKLER['unassigned'], 2)
                    cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RENKLER['unassigned'], 2)
            
            # --- / Döngü Sonu ---

            # Sinyalleri arayüze gönder
            self.update_counts_signal.emit(baret_takan_sayisi, baret_takmayan_sayisi)
            
            # OpenCV (BGR) görüntüsünü QImage (RGB) formatına dönüştür
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            p = convert_to_qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
            self.change_pixmap_signal.emit(p)

        # Döngü bittiğinde kaynakları serbest bırak
        cap.release()
        self.finished_signal.emit()

    def stop(self):
        self.alert_signal.emit("Durdurma sinyali alındı...")
        self._running = False
        self.wait() # Thread'in bitmesini bekle


# --- ANA ARAYÜZ SINIFI (MAIN WINDOW) ---
# --- ANA ARAYÜZ SINIFI (MAIN WINDOW) ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("İş Güvenliği Takip Sistemi (YOLOv8)")
        self.setGeometry(100, 100, 1000, 750)
        
        self.thread = None

        # --- YENİ 'İŞ GÜVENLİĞİ' STİL SAYFASI (QSS) ---
        SAFETY_STYLESHEET = """
            /* Ana pencere ve tüm alt bileşenler için varsayılan stil */
            QWidget {
                background-color: #F4F4F4; /* Açık Gri Ana Arka Plan */
                color: #333333;           /* Koyu Gri Yazı Rengi */
                font-family: 'Segoe UI', Arial, sans-serif;
            }

            /* Video Görüntü Alanı */
            QLabel#videoLabel {
                background-color: #000000;
                border: 2px solid #CCCCCC; /* Nötr Gri Çerçeve */
                border-radius: 5px;
            }

            /* Genel Başlıklar (örn: 'Sistem Logları') */
            QLabel {
                font-size: 13px;
                color: #555555;
                font-weight: bold;
            }

            /* Sayaçlar (QLabel'in üzerine yazar) */
            QLabel#countTakan {
                font-size: 18px;
                font-weight: bold;
                color: #388E3C; /* Koyu Güvenli Yeşil */
                padding: 5px;
            }
            QLabel#countTakmayan {
                font-size: 18px;
                font-weight: bold;
                color: #D32F2F; /* Koyu Tehlike Kırmızısı */
                padding: 5px;
            }
            
            /* Ana Butonlar (Güvenlik Sarısı) */
            QPushButton {
                background-color: #FFB300; /* Ana Güvenlik Sarısı */
                color: #333333;            /* Okunabilirlik için koyu yazı */
                font-weight: bold;
                border: none;
                padding: 10px 15px;
                border-radius: 5px;
                font-size: 14px;
            }
            /* Buton üzerine gelince */
            QPushButton:hover {
                background-color: #FFA000; /* Daha koyu sarı */
            }
            /* Butona basılınca */
            QPushButton:pressed {
                background-color: #FF8F00;
            }
            /* Devre dışı bırakılmış buton */
            QPushButton:disabled {
                background-color: #BDBDBD; /* Devre dışı gri */
                color: #757575;
            }

            /* 'Durdur' Butonu (Tehlike Kırmızısı) */
            QPushButton#stopButton {
                background-color: #D32F2F; /* Koyu Kırmızı */
                color: white;
            }
            QPushButton#stopButton:hover {
                background-color: #B71C1C;
            }
            QPushButton#stopButton:pressed {
                background-color: #9B1A1A;
            }
            
            /* Log Kutusu */
            QTextEdit {
                background-color: #FFFFFF; /* Temiz beyaz */
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                font-family: 'Consolas', 'Courier New', monospace; /* Loglar için sabit genişlikli font */
                font-size: 11px;
                color: #222222;
            }
        """
        # Stil sayfasını ana pencereye uygula
        self.setStyleSheet(SAFETY_STYLESHEET)

        # --- Arayüz Bileşenleri ---
        
        # 1. Video Görüntü Alanı
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setText("Kaynak seçerek işlemi başlatın...")
        self.image_label.setFixedSize(960, 540)
        self.image_label.setObjectName("videoLabel") # QSS'in hedeflemesi için ID

        # 2. Kontrol Butonları
        self.btn_start_webcam = QPushButton("Webcam Başlat (0)")
        self.btn_start_video = QPushButton("Video Dosyası Seç")
        self.btn_stop = QPushButton("Durdur")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setObjectName("stopButton") # QSS için ID
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_start_webcam)
        button_layout.addWidget(self.btn_start_video)
        button_layout.addWidget(self.btn_stop)

        # 3. Sayaç Alanı
        self.label_takan = QLabel("Baret Takan: 0")
        self.label_takan.setObjectName("countTakan") # QSS için ID
        
        self.label_takmayan = QLabel("Baret Takmayan: 0")
        self.label_takmayan.setObjectName("countTakmayan") # QSS için ID

        count_layout = QHBoxLayout()
        count_layout.addStretch(1) # Boşluk ekle (sola yaslamak için)
        count_layout.addWidget(self.label_takan)
        count_layout.addStretch(1) # Araya boşluk ekle
        count_layout.addWidget(self.label_takmayan)
        count_layout.addStretch(1) # Boşluk ekle (sağa yaslamak için)


        # 4. Log/Uyarı Kutusu
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(100)
        self.log_title = QLabel("Sistem Logları ve Uyarılar:") # Ayrı bir etiket olarak ekledik

        # --- Ana Yerleşim (Layout) ---
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addLayout(count_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.log_title)
        main_layout.addWidget(self.log_box)
        main_layout.setSpacing(10) # Bileşenler arası boşluk
        main_layout.setContentsMargins(10, 10, 10, 10) # Pencere kenar boşlukları

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # --- Sinyal Bağlantıları ---
        self.btn_start_webcam.clicked.connect(self.start_webcam)
        self.btn_start_video.clicked.connect(self.start_video_file)
        self.btn_stop.clicked.connect(self.stop_processing)

    # --- Arayüz Fonksiyonları (Slotlar) ---

    def start_webcam(self):
        self.start_processing("0")

    def start_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Video Dosyası Seç", "", "Video Dosyaları (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.start_processing(file_path)

    def start_processing(self, source):
        if self.thread is not None and self.thread.isRunning():
            self.log_alert("Zaten çalışan bir işlem var. Önce durdurun.")
            return

        self.thread = VideoThread(source)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_counts_signal.connect(self.update_counts)
        self.thread.alert_signal.connect(self.log_alert)
        self.thread.finished_signal.connect(self.processing_finished)
        
        self.btn_start_webcam.setEnabled(False)
        self.btn_start_video.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.image_label.setText("Modeller yükleniyor, lütfen bekleyin...")
        
        self.thread.start()

    def stop_processing(self):
        if self.thread:
            self.thread.stop()
        self.processing_finished()

    @pyqtSlot(QImage)
    def update_image(self, qt_image):
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    @pyqtSlot(int, int)
    def update_counts(self, takan_sayisi, takmayan_sayisi):
        self.label_takan.setText(f"Baret Takan: {takan_sayisi}")
        self.label_takmayan.setText(f"Baret Takmayan: {takmayan_sayisi}")

    @pyqtSlot(str)
    def log_alert(self, message):
        self.log_box.append(message)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum()) 

    @pyqtSlot()
    def processing_finished(self):
        self.log_alert("İşlem durduruldu veya bitti.")
        self.btn_start_webcam.setEnabled(True)
        self.btn_start_video.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.image_label.setText("Kaynak seçerek işlemi başlatın...")
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread = None

    def closeEvent(self, event):
        self.stop_processing()
        event.accept()

# --- Uygulamayı Başlat ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # --- YENİ EKLENEN MODERN TEMA KODU ---
    from qt_material import apply_stylesheet
    # 'dark_blue.xml', 'dark_red.xml', 'light_blue.xml' gibi birçok tema var.
    apply_stylesheet(app, theme='dark_blue.xml')
    # --- / YENİ KOD ---

    window = MainWindow()
    window.show()
    sys.exit(app.exec())