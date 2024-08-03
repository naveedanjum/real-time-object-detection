import cv2

class CameraStream:
    def __init__(self, camera_index: int = 0):
        self.camera = cv2.VideoCapture(camera_index)

    def get_frame(self):
        success, frame = self.camera.read()
        if not success:
            raise RuntimeError("Failed to capture frame from camera")
        return frame
