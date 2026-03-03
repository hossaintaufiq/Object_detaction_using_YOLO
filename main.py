import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort

# -------------------------
# Load YOLO Model
# -------------------------
yolo = YOLO("yolov8n.pt")

# -------------------------
# Load Age ONNX Model
# -------------------------
age_session = ort.InferenceSession("age_googlenet.onnx")
input_name = age_session.get_inputs()[0].name

# Age categories (GoogLeNet format)
age_list = [
    "0-2", "4-6", "8-12",
    "15-20", "25-32",
    "38-43", "48-53",
    "60-100"
]

# -------------------------
# Load Face Detector
# -------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------
# Open Camera
# -------------------------
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = yolo(frame, verbose=False)

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])
            if yolo.names[cls] != "person":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw person box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                continue

            gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5
            )

            for (fx, fy, fw, fh) in faces:

                face = person_crop[fy:fy+fh, fx:fx+fw]

                if face.size == 0:
                    continue

                # -------------------------
                # Age Model Preprocessing
                # -------------------------
                face = cv2.resize(face, (224, 224))
                face = face.astype(np.float32) / 255.0

                # HWC -> CHW
                face = np.transpose(face, (2, 0, 1))

                # Add batch dimension
                face = np.expand_dims(face, axis=0)

                # -------------------------
                # Predict Age
                # -------------------------
                pred = age_session.run(None, {input_name: face})[0]
                age_label = age_list[np.argmax(pred)]

                # Draw face box
                cv2.rectangle(
                    frame,
                    (x1+fx, y1+fy),
                    (x1+fx+fw, y1+fy+fh),
                    (255, 0, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Age: {age_label}",
                    (x1+fx, y1+fy-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )

    cv2.imshow("Object + Age Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()