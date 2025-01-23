import cv2
import matplotlib.pyplot as plt



# ścieżki do plików modelu MobileNetSSD
frozen_path = "models/frozen_inference_graph.pb"  # Wagi pretrenowanego modelu
config_path = "models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"  # Architektura sieci
coco_labels = "models/coco.names"
image_path = "data/COCO/test/clip_main_lobby_Tan_mp4-4_jpg.rf.73c9f71a7cc423cd73469e95acfc3894.jpg"

# wczytanie modelu
model = cv2.dnn_DetectionModel(frozen_path, config_path)
classLabels = []

# wczytanie etykiet z pliku coco.names
with open(coco_labels, 'rt') as file:
    classLabels = file.read().rstrip('\n').split('\n')

# konfiguracja modelu
model.setInputSize(320,320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# wczytanie zdjecia
img = cv2.imread(image_path)

# detekcja
class_ids, confidences, boxes = model.detect(img, confThreshold=0.5)

# przechdozenie przez obiekty i rysowanie boxow
for class_id, confidence, box in zip(class_ids, confidences, boxes):
    label = classLabels[class_id - 1]
    if label == "person":
        left, top, width, height = box

        # rysowanie ramki i etykiety
        cv2.rectangle(img, (left, top), (left + width, top + height), (255, 0, 0), 2)
        cv2.putText(img, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# konwersja obrazu z BGR na RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# wyświetlenie obrazu za pomocą Matplotlib
plt.figure(figsize=(10, 6))
plt.imshow(img_rgb)
plt.axis("off")
plt.title("Wyniki wykrywania")
plt.show()