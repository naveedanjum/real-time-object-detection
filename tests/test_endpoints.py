import pytest
from unittest.mock import patch
import numpy as np
import cv2
from app.main import ObjectDetectionApp
from httpx import AsyncClient


def mock_get_frame():
    return np.zeros((640, 480, 3), dtype=np.uint8)


async def mock_gen_frames(*args, **kwargs):
    frame = mock_get_frame()
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@pytest.mark.asyncio
async def test_video_feed():
    app = ObjectDetectionApp()

    with patch('app.services.camera.CameraStream.get_frame', mock_get_frame):
        with patch('app.services.detection.DetectionService.gen_frames', mock_gen_frames):
            async with AsyncClient(app=app.app, base_url="http://test") as client:
                response = await client.get("/video_feed")
                assert response.status_code == 200
                assert response.headers['content-type'].startswith('multipart/x-mixed-replace')
