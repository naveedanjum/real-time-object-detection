import os
import cv2
import json
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from ultralytics import YOLO
from app.services.camera import CameraStream
from app.core.config import DATA_DIR, IMAGES_DIR, MODEL_PATH, CAMERA_INDEX
from app.utils.file_utils import create_directories

class YOLOv8Model:
    def __init__(self, model_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path).to(self.device)

    def predict(self, frame):
        # Ensure frame is in the right format and on the GPU
        frame_resized = cv2.resize(frame, (640, 640))  # Resize to the required dimensions
        frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).unsqueeze(0).float().to(self.device)  # BCHW
        with torch.no_grad():  # Disable gradient calculation for faster inference
            results = self.model(frame_tensor)
        detections = []
        for result in results:
            if hasattr(result, 'boxes'):
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = box.cls[0].item()
                    detections.append([x1, y1, x2, y2, conf, cls])
        return detections

class DetectionService:
    def __init__(self):
        self.model = YOLOv8Model(MODEL_PATH)
        self.data_dir = DATA_DIR
        self.images_dir = IMAGES_DIR
        create_directories(self.data_dir, self.images_dir)
        self.executor = ThreadPoolExecutor()

    async def run_in_executor(self, func, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    async def gen_frames(self):
        stream = CameraStream(CAMERA_INDEX)

        while True:
            frame = await self.run_in_executor(stream.get_frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = await self.run_in_executor(self.model.predict, rgb_frame)
            print(f"Detections: {detections}")  # Debug: print detections

            if detections:
                detection_data = []
                for detection in detections:
                    x1, y1, x2, y2, conf, cls = detection
                    print(f"Detection: {detection}")  # Debug: print each detection
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f'{self.model.model.names[int(cls)]} {conf:.2f}'
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    detection_data.append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "label": self.model.model.names[int(cls)],
                        "confidence": float(conf),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)]
                    })

                await self.run_in_executor(self._save_detections, detection_data)
                await self.run_in_executor(self._save_frame, frame)

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
        print(f"Detections saved to {json_filename}")  # Debug: confirm save

    def _save_frame(self, frame):
        image_filename = os.path.join(self.images_dir, f"frame_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.jpg")
        cv2.imwrite(image_filename, frame)
        print(f"Frame saved to {image_filename}")  # Debug: confirm save

# Function to start the detection service
async def start_detection_service():
    detection_service = DetectionService()
    async for frame in detection_service.gen_frames():
        # You can send the frame to a web client or display it as needed
        pass
