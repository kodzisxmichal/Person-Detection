from ultralytics import YOLO

model_path = "/models/yolov8n.pt"
model = YOLO(model_path)

results = model.train(
  data = "person_config.yaml",
  epochs = 5,
  batch = 16,
  mosaic = False,
  plots = True
)
