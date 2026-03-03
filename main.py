import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort

# Load Models
yolo = YOLO("yolov8n.pt")

# Load Age Model
age_session = ort.InferenceSession("age_googlenet.onnx")

# Age labels (GoogLeNet Age Model)
age_list = [
    "0-2", "4-6", "8-12",
    "15-20", "25-32",
    "38-43", "48-53",
    "60-100"
]

# Webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = yolo(frame, verbose=False)

    for r in results:

        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Draw object box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"{yolo.names[cls]} {conf:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # -------- Age Detection --------
            if yolo.names[cls] == "face":

                try:
                    # Upper part of body = approximate face region
                    face = frame[
                        y1:int(y1 + (y2 - y1) * 0.45),
                        x1:x2
                    ]

                    if face.size > 0:

                        face = cv2.resize(face, (224, 224))
                        face = face.astype(np.float32) / 255.0
                        face = np.expand_dims(face, 0)

                        # Predict age
                        input_name = age_session.get_inputs()[0].name

                        pred = age_session.run(
                            None,
                            {input_name: face}
                        )[0]

                        age_label = age_list[np.argmax(pred)]

                        cv2.putText(
                            frame,
                            f"Age: {age_label}",
                            (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2
                        )

                except:
                    pass

    cv2.imshow("AI Vision System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()