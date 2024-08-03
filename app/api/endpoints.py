from fastapi import APIRouter
from starlette.responses import StreamingResponse
from app.services.detection import DetectionService

router = APIRouter()

detection_service = DetectionService()

@router.get("/video_feed")
async def video_feed():
    return StreamingResponse(detection_service.gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

def create_routes(app):
    app.include_router(router)
