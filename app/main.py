import uvicorn
from fastapi import FastAPI
from app.api.endpoints import create_routes

class ObjectDetectionApp:
    def __init__(self):
        self.app = FastAPI()
        create_routes(self.app)

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        uvicorn.run(self.app, host=host, port=port)