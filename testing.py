from ultralytics import YOLO

# load model from local path
model = YOLO("models/yolov8n.pt")

# run detection on webcam
model.predict(
    source=0,      # 0 = default webcam
    show=True,
    conf=0.5
)
