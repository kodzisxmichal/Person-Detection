import cv2
import ultralytics
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

model_path = "runs/detect/train10/weights/best.pt"
image_path = "data/YOLO/test/images/clip_main_lobby_Tan_mp4-4_jpg.rf.73c9f71a7cc423cd73469e95acfc3894.jpg"


def display_image(image, title="Image"):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

model = ultralytics.YOLO(model_path)

# Odczytanie zdjęcia i przewidywanie obiektów
image = cv2.imread(image_path)
results = model.predict(image, conf=0.5)

data = results[0]  # Pobranie wyników dla pierwszego zdjęcia
annotated_image = data.plot()  # Obraz z ramkami

bbox = data.boxes[0].xyxy[0].tolist()  # Współrzędne bbox: [x_min, y_min, x_max, y_max]


# Wyświetlenie obrazu z adnotacjami
display_image(annotated_image)
