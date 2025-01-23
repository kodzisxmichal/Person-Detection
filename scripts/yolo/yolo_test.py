import cv2
import ultralytics
import matplotlib.pyplot as plt


model_path = "runs/detect/train10/weights/best.pt"
image_path = "data/YOLO/test/images/clip_main_lobby_Tan_mp4-4_jpg.rf.73c9f71a7cc423cd73469e95acfc3894.jpg"


def display_image(image, title="Image"):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

model = ultralytics.YOLO(model_path)

# odczytanie zdjecia i przewidywanie obiektow
image = cv2.imread(image_path)
results = model.predict(image, conf=0.5)

data = results[0]  # pobranie wynikow dla pierwszego zdjecia
annotated_image = data.plot()  # obraz z ramkami

bbox = data.boxes[0].xyxy[0].tolist()  # wspolrzedne bbox: [x_min, y_min, x_max, y_max]

# wyswietlenie obrazu z adnotacjami
display_image(annotated_image)

