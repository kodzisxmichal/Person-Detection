import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from sklearn.metrics import precision_score, recall_score, f1_score

image_path = "data/COCO/test/clip_main_lobby_Tan_mp4-4_jpg.rf.73c9f71a7cc423cd73469e95acfc3894.jpg"

# zamiana obrazu na format rgb
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    return transform(image), image


# Define a function to perform inference
def detect_people(image_tensor, model, threshold=0.5):
    with torch.no_grad(): # wylaczenie gradientow - przyspiesza proces i oszczedza pamiec
        predictions = model([image_tensor])

    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    # filtruje obiekty z label=1 (1 to person) oraz scorem wiekszym od tresholdu
    person_boxes = [box for box, label, score in zip(boxes, labels, scores) if label == 1 and score > threshold]
    return person_boxes

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval()


# rysowanie ramki
def draw_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box.tolist(), outline="red", width=3)
    return image

# preproccesing
image_tensor, original_image = preprocess_image(image_path)

# wykrycie i narysowanie ramki
person_boxes = detect_people(image_tensor, model)
output_image = draw_boxes(original_image, person_boxes)

# wyswietlenie i zapisanie zdjecia
output_image.show()
# output_image.save("output.jpg")
