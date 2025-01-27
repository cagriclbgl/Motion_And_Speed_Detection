#Çağrı Çelebioğlu
import cv2
import numpy as np
from tkinter import Tk, filedialog


# Kullanıcıdan video veya kamera seçimi al
def choose_input_source():
    print("Hareket algılamayı başlatmak için bir seçenek belirleyin:")
    print("1. Kamera ile hareket algıla")
    print("2. Hazır video ile hareket algıla")

    choice = input("Seçiminizi yapın (1 veya 2): ")

    if choice == '1':
        return "camera"
    elif choice == '2':
        return "video"
    else:
        print("Geçersiz seçim! Tekrar deneyin.")
        return choose_input_source()


# Hazır video dosyasını seçme
def choose_video_file():
    root = Tk()
    root.title("Video Seçici")  # Başlık ekleyelim
    file_path = filedialog.askopenfilename(title="Video Dosyasını Seçin",
                                           filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))
    root.destroy()  # Tkinter uygulamasını sonlandır
    return file_path


# Hareketli nesnelerin hızını hesaplama
def calculate_speed(current_position, previous_position, current_time, previous_time, pixels_per_meter):
    if previous_position is None or previous_time is None:
        return None  # İlk karede hız hesaplanmaz
    # Zaman farkı
    time_diff = current_time - previous_time
    # Mesafe hesaplama (Euclidean distance)
    dist = np.linalg.norm(np.array(current_position) - np.array(previous_position))
    # Hız (pixel/saniye)
    speed_px_per_s = dist / time_diff
    # Hızı metre/saniye cinsine çevirmek
    speed_m_per_s = speed_px_per_s * pixels_per_meter
    return speed_m_per_s


# Kamera ile hareket algılama ve hız hesaplama
def motion_detection_camera():
    cap = cv2.VideoCapture(0)  # Varsayılan kamera
    previous_position = None
    previous_time = None
    pixels_per_meter = 0.001  # 1 piksel = 0.001 metre

    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    fgbg = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera görüntüsü alınamadı!")
            break

        fgmask = fgbg.apply(frame)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)  # Nesnenin merkez noktası

            # Hızı hesapla
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # saniye cinsinden
            speed = calculate_speed(center, previous_position, current_time, previous_time, pixels_per_meter)

            if speed is not None:
                cv2.putText(frame, f"Speed: {speed:.3f} m/s", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Yeni pozisyonu ve zamanı kaydet
            previous_position = center
            previous_time = current_time

            # Nesnenin etrafını yeşil renkle çizen dikdörtgen
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Pencereyi yeniden boyutlandırmak için (640x480 boyutunda)
        resized_frame = cv2.resize(frame, (640, 480))

        cv2.imshow("Detected Motion", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Video ile hareket algılama ve hız hesaplama
def motion_detection_video(video_path):
    cap = cv2.VideoCapture(video_path)
    previous_position = None
    previous_time = None
    pixels_per_meter = 0.001  # 1 piksel = 0.001 metre

    if not cap.isOpened():
        print("Video açılamadı!")
        return

    fgbg = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video bitmiştir veya okunamadı!")
            break

        fgmask = fgbg.apply(frame)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)  # Nesnenin merkez noktası

            # Hızı hesapla
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # saniye cinsinden
            speed = calculate_speed(center, previous_position, current_time, previous_time, pixels_per_meter)

            if speed is not None:
                cv2.putText(frame, f"Speed: {speed:.3f} m/s", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Yeni pozisyonu ve zamanı kaydet
            previous_position = center
            previous_time = current_time

            # Nesnenin etrafını yeşil renkle çizen dikdörtgen
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Pencereyi yeniden boyutlandırmak için (640x480 boyutunda)
        resized_frame = cv2.resize(frame, (640, 480))

        cv2.imshow("Detected Motion", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Kullanıcı seçim işlemine göre ilerleme
def start_motion_detection():
    choice = choose_input_source()

    if choice == "camera":
        motion_detection_camera()
    elif choice == "video":
        video_path = choose_video_file()
        if video_path:
            print(f"Video yolu: {video_path}")  # Seçilen video yolunu yazdır
            motion_detection_video(video_path)
        else:
            print("Bir video dosyası seçilmedi!")


if __name__ == "__main__":
    start_motion_detection()
