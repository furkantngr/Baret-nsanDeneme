import sys
import cv2
from ultralytics import YOLO
import time
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QWidget, QPushButton, QHBoxLayout, QTextEdit, 
                             QFileDialog, QFrame, QGridLayout, QGroupBox, QSplitter)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, pyqtSlot, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QIcon
import logging

# --- AYARLAR ---
MODEL_HELMET_PATH = 'best.pt'
MODEL_PERSON_PATH = 'yolov8n.pt'
HELMET_GUVEN_ESIGI = 0.80 
PERSON_GUVEN_ESIGI = 0.25 
UYARI_SURESI = 10  # saniye

# Loglama ayarları
log_format = "%(asctime)s [%(levelname)s] - %(message)s"
formatter = logging.Formatter(log_format)
file_handler = logging.FileHandler("is_guvenligi.log", mode='a', encoding='utf-8')
file_handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(file_handler)

# Renkler (BGR formatında)
RENKLER = {
    'takan': (0, 255, 0),
    'takmayan': (0, 0, 255),
    'unassigned': (128, 128, 128)
}

# --- YARDIMCI FONKSİYONLAR ---
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

# --- VİDEO İŞ PARÇACIĞI ---
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    update_counts_signal = pyqtSignal(int, int)
    alert_signal = pyqtSignal(str, str)  # (message, level)
    finished_signal = pyqtSignal()
    fps_signal = pyqtSignal(float)

    def __init__(self, source):
        super().__init__()
        self.source = source
        self._running = True
        self.ihlal_takip_listesi = {}
        self.HELMET_CLASS_ID = None
        self.frame_count = 0
        self.start_time = time.time()

    def load_models(self):
        try:
            self.alert_signal.emit(f"Kişi modeli yükleniyor...", "INFO")
            self.model_person = YOLO(MODEL_PERSON_PATH)
            
            self.alert_signal.emit(f"Baret modeli yükleniyor...", "INFO")
            self.model_helmet = YOLO(MODEL_HELMET_PATH)
            
            helmet_names = self.model_helmet.names
            for class_id, name in helmet_names.items():
                if name == 'helmet':
                    self.HELMET_CLASS_ID = class_id
                    break
            
            if self.HELMET_CLASS_ID is None:
                self.alert_signal.emit(f"HATA: 'helmet' sınıfı bulunamadı!", "ERROR")
                return False
            
            self.alert_signal.emit(f"Modeller başarıyla yüklendi", "SUCCESS")
            return True
        except Exception as e:
            self.alert_signal.emit(f"Model yükleme hatası: {e}", "ERROR")
            return False

    def run(self):
        if not self.load_models():
            self.finished_signal.emit()
            return

        if self.source.isdigit():
            cap = cv2.VideoCapture(int(self.source))
        else:
            cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            self.alert_signal.emit(f"Kaynak açılamadı: {self.source}", "ERROR")
            self.finished_signal.emit()
            return
            
        self.alert_signal.emit(f"İzleme başlatıldı", "SUCCESS")
        fps_start = time.time()
        fps_counter = 0

        while self._running:
            success, frame = cap.read()
            if not success:
                self.alert_signal.emit("Video akışı sonlandı", "WARNING")
                break
            
            # FPS hesaplama
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps = 30 / (time.time() - fps_start)
                self.fps_signal.emit(fps)
                fps_start = time.time()

            # Model çalıştırma
            results_person = self.model_person.track(frame, persist=True, classes=[0], 
                                                     conf=PERSON_GUVEN_ESIGI, verbose=False)
            results_helmet = self.model_helmet.track(frame, persist=True, classes=[self.HELMET_CLASS_ID], 
                                                     conf=HELMET_GUVEN_ESIGI, verbose=False)
            
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
                    label = f"ID {person_id}: GUVENLI"
                    cv2.rectangle(frame, (matched_helmet_bbox[0], matched_helmet_bbox[1]), 
                                (matched_helmet_bbox[2], matched_helmet_bbox[3]), RENKLER['takan'], 3)
                    cv2.putText(frame, label, (matched_helmet_bbox[0], matched_helmet_bbox[1] - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, RENKLER['takan'], 2)
                else:
                    baret_takmayan_sayisi += 1
                    current_frame_person_ids_no_helmet.add(person_id)
                    label = f"ID {person_id}: TEHLIKE!"
                    cv2.rectangle(frame, (person_bbox[0], person_bbox[1]), 
                                (person_bbox[2], person_bbox[3]), RENKLER['takmayan'], 3)
                    cv2.putText(frame, label, (person_bbox[0], person_bbox[1] - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, RENKLER['takmayan'], 2)

                    if person_id not in self.ihlal_takip_listesi:
                        self.ihlal_takip_listesi[person_id] = {'start_time': time.time(), 'warned': False}
                    else:
                        data = self.ihlal_takip_listesi[person_id]
                        gecen_sure = time.time() - data['start_time']
                        
                        if gecen_sure > UYARI_SURESI and not data['warned']:
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            self.alert_signal.emit(
                                f"KISI ID {person_id} - {UYARI_SURESI} saniyedir baret takmiyor!", 
                                "CRITICAL"
                            )
                            data['warned'] = True
            
            # İhlal listesi temizliği
            for person_id in list(self.ihlal_takip_listesi.keys()):
                if person_id not in current_frame_person_ids_no_helmet:
                    if self.ihlal_takip_listesi[person_id]['warned']:
                        self.alert_signal.emit(f"KISI ID {person_id} - İhlal durumu düzeltildi", "INFO")
                    del self.ihlal_takip_listesi[person_id]
            
            # İlişkisiz baretler
            for helmet in helmets:
                if helmet['id'] not in drawn_helmet_ids:
                    bbox = helmet['bbox']
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                                RENKLER['unassigned'], 2)

            # Sayaçları güncelle
            self.update_counts_signal.emit(baret_takan_sayisi, baret_takmayan_sayisi)
            
            # Görüntüyü QImage'e çevir
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.change_pixmap_signal.emit(convert_to_qt_format)

        cap.release()
        self.finished_signal.emit()

    def stop(self):
        self._running = False
        self.wait()

# --- ANA ARAYÜZ ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("İş Sağlığı ve Güvenliği - Baret Takip Sistemi v2.0")
        self.setGeometry(50, 50, 1600, 900)
        self.thread = None
        self.total_violations = 0
        self.session_start = None
        
        self.setup_ui()
        self.apply_stylesheet()
        
        # Saat güncelleyici
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)
        self.update_clock()

    def setup_ui(self):
        # Ana widget ve layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Sol panel - Video ve kontroller
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)

        # Başlık bölümü
        header_frame = QFrame()
        header_frame.setObjectName("headerFrame")
        header_layout = QVBoxLayout(header_frame)
        
        subtitle_label = QLabel("Baret Takip ve İzleme Sistemi")
        subtitle_label.setObjectName("subtitleLabel")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        header_layout.addWidget(subtitle_label)
        
        # Video görüntü alanı
        video_frame = QFrame()
        video_frame.setObjectName("videoFrame")
        video_layout = QVBoxLayout(video_frame)
        
        self.image_label = QLabel()
        self.image_label.setObjectName("videoLabel")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setText("📹\n\nSistem Hazır\n\nİzlemeyi başlatmak için kaynak seçin")
        self.image_label.setMinimumSize(960, 540)
        self.image_label.setScaledContents(False)
        
        video_layout.addWidget(self.image_label)
        
        # FPS ve durum göstergesi
        status_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setObjectName("statusLabel")
        self.status_label = QLabel("● Beklemede")
        self.status_label.setObjectName("statusLabel")
        status_layout.addWidget(self.fps_label)
        status_layout.addStretch()
        status_layout.addWidget(self.status_label)
        video_layout.addLayout(status_layout)

        # Kontrol butonları
        control_frame = QFrame()
        control_frame.setObjectName("controlFrame")
        control_layout = QGridLayout(control_frame)
        control_layout.setSpacing(10)
        
        self.btn_webcam = QPushButton("🎥 KAMERA BAŞLAT")
        self.btn_webcam.setObjectName("startButton")
        self.btn_webcam.setMinimumHeight(50)
        
        self.btn_video = QPushButton("📁 VİDEO DOSYASI")
        self.btn_video.setObjectName("startButton")
        self.btn_video.setMinimumHeight(50)
        
        self.btn_stop = QPushButton("⏹ DURDUR")
        self.btn_stop.setObjectName("stopButton")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setMinimumHeight(50)
        
        control_layout.addWidget(self.btn_webcam, 0, 0)
        control_layout.addWidget(self.btn_video, 0, 1)
        control_layout.addWidget(self.btn_stop, 1, 0, 1, 2)

        left_layout.addWidget(header_frame)
        left_layout.addWidget(video_frame)
        left_layout.addWidget(control_frame)

        # Sağ panel - İstatistikler ve loglar
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(15)
        
        # Saat ve tarih
        self.clock_label = QLabel()
        self.clock_label.setObjectName("clockLabel")
        self.clock_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.clock_label)

        # Anlık durum kartları
        stats_group = QGroupBox("ANLIK DURUM")
        stats_group.setObjectName("statsGroup")
        stats_layout = QVBoxLayout(stats_group)
        
        # Güvenli sayaç
        safe_card = QFrame()
        safe_card.setObjectName("safeCard")
        safe_layout = QVBoxLayout(safe_card)
        self.label_takan = QLabel("0")
        self.label_takan.setObjectName("safeCount")
        self.label_takan.setAlignment(Qt.AlignmentFlag.AlignCenter)
        safe_text = QLabel("✓ Güvenli Personel")
        safe_text.setObjectName("cardLabel")
        safe_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        safe_layout.addWidget(self.label_takan)
        safe_layout.addWidget(safe_text)
        
        # Tehlikeli sayaç
        danger_card = QFrame()
        danger_card.setObjectName("dangerCard")
        danger_layout = QVBoxLayout(danger_card)
        self.label_takmayan = QLabel("0")
        self.label_takmayan.setObjectName("dangerCount")
        self.label_takmayan.setAlignment(Qt.AlignmentFlag.AlignCenter)
        danger_text = QLabel("⚠ Tehlikede Personel")
        danger_text.setObjectName("cardLabel")
        danger_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        danger_layout.addWidget(self.label_takmayan)
        danger_layout.addWidget(danger_text)
        
        stats_layout.addWidget(safe_card)
        stats_layout.addWidget(danger_card)
        right_layout.addWidget(stats_group)

        # Toplam istatistikler
        summary_group = QGroupBox("OTURUM İSTATİSTİKLERİ")
        summary_group.setObjectName("statsGroup")
        summary_layout = QVBoxLayout(summary_group)
        
        self.violation_label = QLabel("Toplam İhlal: 0")
        self.violation_label.setObjectName("summaryLabel")
        self.session_label = QLabel("Oturum Süresi: 00:00:00")
        self.session_label.setObjectName("summaryLabel")
        
        summary_layout.addWidget(self.violation_label)
        summary_layout.addWidget(self.session_label)
        right_layout.addWidget(summary_group)

        # Log konsolu
        log_group = QGroupBox("SİSTEM KAYITLARI")
        log_group.setObjectName("statsGroup")
        log_layout = QVBoxLayout(log_group)
        
        self.log_box = QTextEdit()
        self.log_box.setObjectName("logBox")
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(250)
        
        log_layout.addWidget(self.log_box)
        right_layout.addWidget(log_group)
        right_layout.addStretch()

        # Ana layout'a panelleri ekle
        main_layout.addWidget(left_panel, 7)
        main_layout.addWidget(right_panel, 3)

        # Sinyal bağlantıları
        self.btn_webcam.clicked.connect(self.start_webcam)
        self.btn_video.clicked.connect(self.start_video_file)
        self.btn_stop.clicked.connect(self.stop_processing)

    def apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            
            QWidget {
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }
            
            
            #titleLabel {
                font-size: 20px;
                font-weight: bold;
                color: #1a1a1a;
                letter-spacing: 1px;
            }
            
            #subtitleLabel {
                font-size: 20px;
                color: #ffffff;
                font-weight: 500;
                margin-top: 2px;
                background-color: #2a2a2a;
                border-radius: 8px;
                padding: 15px;
            }
            
            #videoFrame {
                background-color: #2a2a2a;
                border: 3px solid #FFB300;
                border-radius: 10px;
                padding: 10px;
            }
            
            #videoLabel {
                background-color: #000000;
                border: 2px solid #404040;
                border-radius: 5px;
                font-size: 24px;
                color: #808080;
                padding: 20px;
            }
            
            #statusLabel {
                color: #b0b0b0;
                font-size: 12px;
                padding: 5px;
            }
            
            #controlFrame {
                background-color: #2a2a2a;
                border-radius: 10px;
                padding: 15px;
            }
            
            #startButton {
                background-color: #FFB300;
                color: #1a1a1a;
                font-weight: bold;
                font-size: 14px;
                border: none;
                border-radius: 8px;
                padding: 10px;
            }
            
            #startButton:hover {
                background-color: #FFA000;
            }
            
            #startButton:pressed {
                background-color: #FF8F00;
            }
            
            #startButton:disabled {
                background-color: #404040;
                color: #808080;
            }
            
            #stopButton {
                background-color: #D32F2F;
                color: white;
                font-weight: bold;
                font-size: 14px;
                border: none;
                border-radius: 8px;
                padding: 10px;
            }
            
            #stopButton:hover {
                background-color: #B71C1C;
            }
            
            #stopButton:pressed {
                background-color: #9B1A1A;
            }
            
            #stopButton:disabled {
                background-color: #404040;
                color: #808080;
            }
            
            #clockLabel {
                font-size: 22px;
                font-weight: bold;
                color: #FFB300;
                background-color: #2a2a2a;
                border-radius: 8px;
                padding: 15px;
            }
            
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                color: #FFB300;
                border: 2px solid #404040;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 15px;
                padding: 0 5px;
                background-color: #1a1a1a;
            }
            
            #safeCard {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2E7D32, stop:1 #1B5E20);
                border-radius: 10px;
                padding: 20px;
                border: 2px solid #4CAF50;
            }
            
            #dangerCard {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #C62828, stop:1 #8E0000);
                border-radius: 10px;
                padding: 20px;
                border: 2px solid #F44336;
            }
            
            #safeCount, #dangerCount {
                font-size: 56px;
                font-weight: bold;
                color: white;
            }
            
            #cardLabel {
                font-size: 14px;
                color: white;
                font-weight: 600;
            }
            
            #summaryLabel {
                color: #e0e0e0;
                font-size: 13px;
                font-weight: normal;
                padding: 8px;
                background-color: #2a2a2a;
                border-radius: 5px;
                margin: 3px;
            }
            
            #logBox {
                background-color: #0a0a0a;
                color: #00ff00;
                border: 2px solid #404040;
                border-radius: 8px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                padding: 10px;
            }
        """)

    def update_clock(self):
        now = datetime.now()
        self.clock_label.setText(now.strftime("%H:%M:%S\n%d.%m.%Y"))
        
        if self.session_start:
            elapsed = int((datetime.now() - self.session_start).total_seconds())
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = elapsed % 60
            self.session_label.setText(f"Oturum Süresi: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def start_webcam(self):
        self.start_processing("0")

    def start_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Video Dosyası Seç", "", 
            "Video Dosyaları (*.mp4 *.avi *.mov *.mkv);;Tüm Dosyalar (*.*)"
        )
        if file_path:
            self.start_processing(file_path)

    def start_processing(self, source):
        if self.thread is not None and self.thread.isRunning():
            self.log_message("Sistem zaten çalışıyor!", "WARNING")
            return

        self.thread = VideoThread(source)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_counts_signal.connect(self.update_counts)
        self.thread.alert_signal.connect(self.log_message)
        self.thread.finished_signal.connect(self.processing_finished)
        self.thread.fps_signal.connect(self.update_fps)
        
        self.btn_webcam.setEnabled(False)
        self.btn_video.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_label.setText("● İşleniyor")
        self.status_label.setStyleSheet("color: #4CAF50;")
        self.image_label.setText("⚙\n\nModeller yükleniyor...\n\nLütfen bekleyin")
        
        self.session_start = datetime.now()
        self.total_violations = 0
        self.thread.start()

    def stop_processing(self):
        if self.thread:
            self.thread.stop()
            self.log_message("Durdurma komutu gönderildi", "INFO")

    @pyqtSlot(QImage)
    def update_image(self, qt_image):
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    @pyqtSlot(int, int)
    def update_counts(self, takan, takmayan):
        self.label_takan.setText(str(takan))
        self.label_takmayan.setText(str(takmayan))

    @pyqtSlot(float)
    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    @pyqtSlot(str, str)
    def log_message(self, message, level):
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Renk kodları
        colors = {
            "INFO": "#00ff00",
            "SUCCESS": "#00ff00",
            "WARNING": "#FFB300",
            "ERROR": "#ff0000",
            "CRITICAL": "#ff0000"
        }
        
        symbols = {
            "INFO": "ℹ",
            "SUCCESS": "✓",
            "WARNING": "⚠",
            "ERROR": "✗",
            "CRITICAL": "🚨"
        }
        
        color = colors.get(level, "#00ff00")
        symbol = symbols.get(level, "•")
        
        formatted_msg = f'<span style="color:{color}">[{timestamp}] {symbol} {message}</span>'
        self.log_box.append(formatted_msg)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())
        
        # Loglama
        if level == "ERROR" or level == "CRITICAL":
            logging.error(message)
            if level == "CRITICAL":
                self.total_violations += 1
                self.violation_label.setText(f"Toplam İhlal: {self.total_violations}")
        elif level == "WARNING":
            logging.warning(message)
        else:
            logging.info(message)

    @pyqtSlot()
    def processing_finished(self):
        self.log_message("İzleme durduruldu", "INFO")
        self.btn_webcam.setEnabled(True)
        self.btn_video.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText("● Beklemede")
        self.status_label.setStyleSheet("color: #808080;")
        self.image_label.clear()
        self.image_label.setText("📹\n\nSistem Hazır\n\nİzlemeyi başlatmak için kaynak seçin")
        self.fps_label.setText("FPS: --")
        self.session_start = None
        
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread = None

    def closeEvent(self, event):
        """Uygulama kapatılırken thread'i düzgün sonlandır"""
        if self.thread and self.thread.isRunning():
            self.thread.stop()
        event.accept()

# --- UYGULAMA BAŞLATMA ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern görünüm için
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())