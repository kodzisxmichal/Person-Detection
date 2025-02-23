import cv2
import matplotlib.pyplot as plt
import json
import os
import glob


# Paths
frozen_path = "models/frozen_inference_graph.pb"  # Weights
config_path = "models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"  # Network Architecture
coco_labels = "models/coco.names"
images_path = "data/COCO/test/"

tp_all, fp_all, fn_all = 0, 0, 0

# Loads ground truths labels for a specific image from COCO json file
def load_coco_labels(annotation_path, image_name):
    with open(annotation_path, "r") as file:
        data = json.load(file)

    # Find image_id for the given image
    image_id = None
    for img in data["images"]:
        if img["file_name"] == image_name:
            image_id = img["id"]
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


# Loading model
model = cv2.dnn_DetectionModel(frozen_path, config_path)
classLabels = []

# Reading classes from coco file
with open(coco_labels, 'rt') as file:
    classLabels = file.read().rstrip('\n').split('\n')

# Model configuration
model.setInputSize(320,320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Collect all images path into list
images = glob.glob(os.path.join(images_path, '*.jpg'))
print(len(images))

for image_path in images:

    # Divides paths into directories and filenames
    image_directory = os.path.dirname(image_path)
    image_name = os.path.basename(image_path)

    # Loading images
    img = cv2.imread(image_path)

    # Detection
    class_ids, confidences, boxes = model.detect(img, confThreshold=0.5)

    # Iterating boxes and drawing bounding box
    for class_id, confidence, box in zip(class_ids, confidences, boxes):
        label = classLabels[class_id - 1]
        if label == "person":
            left, top, width, height = box

            # Drawing bounding box
            cv2.rectangle(img, (left, top), (left + width, top + height), (255, 0, 0), 2)
            cv2.putText(img, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Image covertion from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Diplaying image Matplotlib
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()

    # Paths
    annotation_path = "data/COCO/test/_annotations.coco.json"
    image_name = os.path.basename(image_path)  # Extract filename from path

    # Load ground truth labels
    gt_boxes = load_coco_labels(annotation_path, image_name)

    # Perform detection using OpenCV
    class_ids, confidences, boxes = model.detect(img, confThreshold=0.5)

    # Convert detections to XYXY format
    pred_boxes = [(left, top, left + width, top + height) for box in boxes for left, top, width, height in [box]]

    # Match predictions with ground truth
    tp, fp, fn = match_predictions(pred_boxes, gt_boxes)

    tp_all += tp
    fp_all += fp
    fn_all += fn

# Compute metrics
precision, recall, f1 = calculate_metrics(tp, fp, fn)

print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

