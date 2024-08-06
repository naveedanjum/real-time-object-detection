import numpy as np
from app.services.detection import YOLOv8Model, DetectionService

def test_yolov8_model():
    model = YOLOv8Model(model_path="yolov10n.pt")
    frame = np.zeros((640, 480, 3), dtype=np.uint8)
    detections = model.predict(frame)
    assert isinstance(detections, list)

def test_detection_service():
    service = DetectionService()
    frame = np.zeros((640, 480, 3), dtype=np.uint8)
    detections = service.model.predict(frame)
    assert isinstance(detections, list)
