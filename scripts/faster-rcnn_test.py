import torch
import os
import json
import glob
from torchvision import models, transforms
from PIL import Image, ImageDraw
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

# Paths
images_path = "data/COCO/test/"
annotation_path = "data/COCO/test/_annotations.coco.json"

tp_all, fp_all, fn_all = 0, 0, 0

# Converting image to RGB format
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    return transform(image), image


# Detecting people
def detect_people(image_tensor, model, threshold=0.5):
    with torch.no_grad(): # Turns of gradient - speeds up the process
        predictions = model([image_tensor])

    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    # Filters objects with label=1 (1 stands for person) and with score than treshold
    person_boxes = [box for box, label, score in zip(boxes, labels, scores) if label == 1 and score > threshold]
    return person_boxes

# Drawing bounding box
def draw_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box.tolist(), outline="red", width=3)
    return image

## Metrics

# Loads ground truths labels for a specific image from COCO json file
def load_coco_labels(annotation_path, image_name):
    with open(annotation_path, "r") as file:
        data = json.load(file)

    # Find image_id for the given image
    image_id = None
    for img in data["images"]:
        if img["file_name"] == image_name:
            image_id = img["id"]
            width, height = img["width"], img["height"]
            break

    # Extract bounding boxes for the image
    gt_boxes = []
    for ann in data["annotations"]:
        if ann["image_id"] == image_id and ann["category_id"] == 1:  # Only 'person' category
            x, y, w, h = ann["bbox"]
            x_max = x + w
            y_max = y + h
            gt_boxes.append((x, y, x_max, y_max))  # Convert to XYXY format

    return gt_boxes

# Compute IoU between two bounding boxes in (x_min, y_min, x_max, y_max) format.
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area_box1 + area_box2 - intersection
    return intersection / union if union > 0 else 0

# Checks and sets if the box is correctly placed
def match_predictions(pred_boxes, gt_boxes, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0
    matched_gt = set()

    for pred_box in pred_boxes:
        best_iou = 0
        best_gt = None

        for i, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt = i

        if best_iou >= iou_threshold and best_gt is not None:
            tp += 1
            matched_gt.add(best_gt)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    return tp, fp, fn

def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

# Collect all images path into list
images = glob.glob(os.path.join(images_path, '*.jpg'))
print(len(images))

for image_path in images:

    # Divides paths into directories and filenames
    image_directory = os.path.dirname(image_path)
    image_name = os.path.basename(image_path)

    # Preproccesing
    image_tensor, original_image = preprocess_image((image_directory + "/" + image_name))

    # Detecting and drawing bounding box
    person_boxes = detect_people(image_tensor, model)
    output_image = draw_boxes(original_image, person_boxes)

    # Display image
    output_image.show()

    # Load ground truth boxes
    gt_boxes = load_coco_labels(annotation_path, image_name)

    # Run inference
    person_boxes = detect_people(image_tensor, model)

    # Convert prediction format to match GT (convert tensor to list)
    pred_boxes = [box.tolist() for box in person_boxes]

    # Match predictions with ground truth
    tp, fp, fn = match_predictions(pred_boxes, gt_boxes)

    tp_all += tp
    fp_all += fp
    fn_all += fn

# Calculate precision, recall, and F1-score
precision, recall, f1 = calculate_metrics(tp_all, fp_all, fn_all)

print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")