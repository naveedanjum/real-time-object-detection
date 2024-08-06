import cv2
import numpy as np
from app.services.camera import CameraStream

def test_camera_stream():
    camera = CameraStream(camera_index=0)
    frame = camera.get_frame()
    assert frame is not None
    assert isinstance(frame, np.ndarray)
