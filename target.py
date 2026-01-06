import time
import cv2
from ultralytics import YOLO

model = YOLO("models/farmdetector.pt")

cap = cv2.VideoCapture(0)

TARGET_CLASS = "pig"
DETECTION_TIME = 3.0  # seconds

pig_start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.3, verbose=False)
    boxes = results[0].boxes
    names = model.names

    pig_detected = False

    if boxes is not None:
        for cls in boxes.cls:
            if names[int(cls)] == TARGET_CLASS:
                pig_detected = True
                break

    current_time = time.time()

    if pig_detected:
        if pig_start_time is None:
            pig_start_time = current_time  # start timer

        elif current_time - pig_start_time >= DETECTION_TIME:
            print("hi")

            # 🔁 RESET SYSTEM (continue searching)
            pig_start_time = None
            time.sleep(1)  # optional cooldown to avoid instant retrigger

    else:
        pig_start_time = None  # pig lost → reset timer

    annotated = results[0].plot()
    cv2.imshow("YOLO Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
