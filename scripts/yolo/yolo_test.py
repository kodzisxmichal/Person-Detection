import cv2
import ultralytics
import glob
import os
import matplotlib.pyplot as plt

model_path = "runs/detect/train10/weights/best.pt"

# Paths
images_path = "data/YOLO/test/images/"
labels_path = "data/YOLO/test/labels/"

tp_all, fp_all, fn_all = 0, 0, 0

def display_image(image, title="Image"):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

def yolo_to_xyxy(label_path, image_shape):
    h, w = image_shape[:2]
    bboxes = []

    with open(label_path, "r") as f:
        for line in f:
            data = line.split()
            class_id = int(data[0])
            x_center, y_center, width, height = map(float, data[1:])

            # Convert to absolute pixel coordinates
            x_min = int((x_center - width / 2) * w)
            y_min = int((y_center - height / 2) * h)
            x_max = int((x_center + width / 2) * w)
            y_max = int((y_center + height / 2) * h)

            bboxes.append((class_id, x_min, y_min, x_max, y_max))

    return bboxes

def calculate_iou(box1, box2):
    # box1, box2 in format (x_min, y_min, x_max, y_max)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area_box1 + area_box2 - intersection
    return intersection / union if union > 0 else 0


def match_predictions(pred_boxes, gt_boxes, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0
    matched_gt = set()

    for pred in pred_boxes:
        pred_class, *pred_box = pred
        best_iou = 0
        best_gt = None

        for i, gt in enumerate(gt_boxes):
            gt_class, *gt_box = gt
            if pred_class == gt_class:  # Match only same class
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

model = ultralytics.YOLO(model_path)

# Collect all images path into list
images = glob.glob(os.path.join(images_path, '*.jpg'))
print(len(images))

for image_path in images:

    # Takes image name from path
    image_name = os.path.basename(image_path)
    # Takes label path and adds image_name with changed .jpg to .txt
    label_path = labels_path + image_name.replace(".jpg", ".txt")

    # Reading images and detecting objects
    image = cv2.imread(image_path)
    results = model.predict(image, conf=0.5)

    data = results[0] # Gathering results for first photo
    annotated_image = data.plot()  # Images with annotations

    bbox = data.boxes[0].xyxy[0].tolist()  # Bounding box: [x_min, y_min, x_max, y_max]

    # Displaying image with annotations
    display_image(annotated_image)

    # Load ground truth
    gt_boxes = yolo_to_xyxy(label_path, image.shape)

    # Load predictions
    pred_boxes = [
        (int(data.boxes.cls[i]), *data.boxes.xyxy[i].tolist())
        for i in range(len(data.boxes))
    ]

    # Match and calculate scores
    tp, fp, fn = match_predictions(pred_boxes, gt_boxes)
    precision, recall, f1 = calculate_metrics(tp, fp, fn)

print("Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

