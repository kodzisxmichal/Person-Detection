import os
import numpy as np
import cv2
import json

# ścieżki wejściowe i wyjściowe
input_dir = "data/COCO"
output_dir = "data/YOLO"

splits = ["train", "valid", "test"]

for split in splits:
    images_output_dir = os.path.join(output_dir, split, "images")
    labels_output_dir = os.path.join(output_dir, split, "labels")

    # tworzenie folderów wyjściowych
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)

    def save_yolo_format(input_dir, split, images_output_dir, labels_output_dir):
        annotations_path = os.path.join(input_dir, split, "_annotations.coco.json")

        with open(annotations_path, "r") as f:
            coco_data = json.load(f)

        images_info = {image["id"]: image for image in coco_data["images"]}
        annotations = coco_data["annotations"]

        for annotation in annotations:
            image_id = annotation["image_id"]
            image_info = images_info[image_id]

            # ścieżki do obrazów i etykiet
            image_filename = image_info["file_name"]
            image_path = os.path.join(input_dir, split, image_filename)

            # wczytanie obrazu
            image = cv2.imread(image_path)

            height, width, _ = image.shape

            # kopiowanie obrazu do katalogu wyjściowego
            output_image_path = os.path.join(images_output_dir, image_filename)
            cv2.imwrite(output_image_path, image)

            # konwersja i zapis etykiet w formacie YOLO
            bbox = annotation["bbox"]
            x_min, y_min, bbox_width, bbox_height = bbox
            x_max, y_max = x_min + bbox_width, y_min + bbox_height

            x_center = ((x_min + x_max) / 2) / width
            y_center = ((y_min + y_max) / 2) / height
            norm_width = bbox_width / width
            norm_height = bbox_height / height

            class_id = annotation["category_id"] - 1  # YOLO indexuje klasy od 0

            label_filename = os.path.splitext(image_filename)[0] + ".txt"
            output_label_path = os.path.join(labels_output_dir, label_filename)

            with open(output_label_path, "a") as f:
                f.write(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}\n")

    save_yolo_format(input_dir, split, images_output_dir, labels_output_dir)
