import os
import cv2
import json
from datetime import datetime
from ultralytics import YOLO
from app.services.camera import CameraStream
from app.core.config import DATA_DIR, IMAGES_DIR, MODEL_PATH, CAMERA_INDEX
from app.utils.file_utils import create_directories

class YOLOv8Model:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict(self, frame):
        results = self.model(frame)
        return results.xyxy[0].numpy().tolist()

class DetectionService:
    def __init__(self):
        self.model = YOLOv8Model(MODEL_PATH)
        self.data_dir = DATA_DIR
        self.images_dir = IMAGES_DIR
        create_directories(self.data_dir, self.images_dir)

    def gen_frames(self):
        stream = CameraStream(CAMERA_INDEX)

        while True:
            frame = stream.get_frame()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = self.model.predict(rgb_frame)

            detection_data = []
            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f'{self.model.model.names[int(cls)]} {conf:.2f}'
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                detection_data.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "label": self.model.model.names[int(cls)],
                    "confidence": float(conf),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })

            self._save_detections(detection_data)
            self._save_frame(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def _save_detections(self, detections):
        current_date = datetime.utcnow().strftime('%Y-%m-%d')
        json_filename = os.path.join(self.data_dir, f"detections-{current_date}.json")

        with open(json_filename, 'a') as f:
            for detection in detections:
                f.write(json.dumps(detection) + '\n')

    def _save_frame(self, frame):
        image_filename = os.path.join(self.images_dir, f"frame_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.jpg")
        cv2.imwrite(image_filename, frame)
