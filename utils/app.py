import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model anomaly detection
model = load_model("./model/anomaly_detection_model.h5")

# Daftar nama kelas
class_labels = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", 
    "Normal Videos", "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"
]

def preprocess_image(image):
    image = cv2.resize(image, (64, 64))  # Sesuaikan ukuran dengan model
    image = image / 255.0  # Normalisasi
    image = np.expand_dims(image, axis=0)  # Tambahkan batch dimension
    return image

def predict_anomaly(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_id = np.argmax(prediction)  # Menyesuaikan untuk 14 kelas
    return class_id, class_labels[class_id]

# 1. Pengujian dengan webcam
def test_with_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        class_id, label = predict_anomaly(frame)
        
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Webcam Anomaly Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# 2. Pengujian dengan path gambar
def test_with_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Gagal membaca gambar.")
        return
    
    class_id, label = predict_anomaly(image)
    
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Image Anomaly Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# test_with_webcam() 
test_with_image("./datasets/Test/Vandalism007_x264_0.png") 