import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort

# -------------------------
# Load YOLO Model
# -------------------------
yolo = YOLO("yolov8n.pt")

# -------------------------
# Load Age Model
# -------------------------
age_session = ort.InferenceSession("age_googlenet.onnx")
age_input = age_session.get_inputs()[0].name

age_list = [
    "0-2", "4-6", "8-12",
    "15-20", "25-32",
    "38-43", "48-53",
    "60-100"
]

# -------------------------
# Load Emotion Model (FER+ opset 8)
# -------------------------
emotion_session = ort.InferenceSession("emotion-ferplus-8.onnx")
emotion_input = emotion_session.get_inputs()[0].name

emotion_labels = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt"
]

# -------------------------
# Face Detector
# -------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

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

            label = f"{yolo.names[cls]} {conf:.2f}"

            # Draw object box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

            # -------- Only process PERSON --------
            if yolo.names[cls] == "person":

                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue

                gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(
                    gray, 1.3, 5
                )

                for (fx, fy, fw, fh) in faces:

                    face = person_crop[fy:fy+fh, fx:fx+fw]
                    if face.size == 0:
                        continue

                    # ---------------- AGE ----------------
                    age_face = cv2.resize(face, (224, 224))
                    age_face = age_face.astype(np.float32) / 255.0
                    age_face = np.transpose(age_face, (2, 0, 1))
                    age_face = np.expand_dims(age_face, axis=0)

                    age_pred = age_session.run(
                        None, {age_input: age_face}
                    )[0]

                    age_label = age_list[np.argmax(age_pred)]

                    # ---------------- EMOTION ----------------
                    emo_face = cv2.resize(face, (64, 64))
                    # Convert to grayscale
                    emo_face = cv2.cvtColor(emo_face, cv2.COLOR_BGR2GRAY)

                    # Convert to float32
                    emo_face = emo_face.astype(np.float32)
                    # IMPORTANT: FER+ usually needs mean normalization
                    emo_face = (emo_face - 127.5) / 128.0

                    emo_face = np.expand_dims(emo_face, axis=0)
                    emo_face = np.expand_dims(emo_face, axis=0)

                    emotion_pred = emotion_session.run(
                       None, {emotion_input: emo_face}
                    )[0]

                    # Get highest confidence
                    emotion_index = np.argmax(emotion_pred)
                    emotion_conf = np.max(emotion_pred)

                    emotion_label = emotion_labels[emotion_index]

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
                        f"{age_label} | {emotion_label}",
                        (x1+fx, y1+fy-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2
                    )

    cv2.imshow("Object + Age + Emotion", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()