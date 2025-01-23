# System detekcji osób na zdjęciach

## Opis projektu
Projekt opiera się na detekcji osób na zdjęciach przy użyciu trzech różnych modeli: YOLOv8, Mobilenet SSD, i Faster R-CNN.
Celem jest nauka AI 

## Wymagania
- Python 3.8+
- Zainstalowanie PyTroch
- Zainstalowane zależności z `requirements.txt`

## Uruchomienie
1. Skopiuj repozytorium.
2. Zainstaluj PyTorch:
   ```bash 
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
3. Zainstaluj wymagane biblioteki:
   ```bash
   pip install -r requirements.txt
   
4. Yolov8
   ```bash
   python.exe scripts/yolo_test.py
5. Mobilenet SSD
   ```bash
   python.exe scripts/faster-rcnn_test.py
6. Faster R-CNN
   ```bash
   python.exe scripts/yolo_test.py

## Źródła
- https://roboflow.com/
- https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
- https://github.com/chuanqi305/MobileNet-SSD/tree/master
- https://github.com/ultralytics/ultralytics/tree/main
- https://www.youtube.com/watch?v=e-tfaEK9sFs&t=309s
- https://github.com/AlexeyAB/darknet/tree/master