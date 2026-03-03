import cv2
import insightface

class AIAssistant:

    def __init__(self):
        self.model = insightface.app.FaceAnalysis()
        self.model.prepare(ctx_id=0)

    def detect(self, frame):

        faces = self.model.get(frame)

        for face in faces:

            x1, y1, x2, y2 = map(int, face.bbox)

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            # Gender & Age if available
            if hasattr(face, 'age'):
                cv2.putText(
                    frame,
                    f"Age: {int(face.age)}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )

        return frame