from __future__ import annotations

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
import logging

from app.core.logging import setup_logging
from app.services.inference import InferenceEngine
from app.services.video_jobs import VideoJobStore
from app.services.websocket import handle_stream


setup_logging()
logger = logging.getLogger("app.startup")

app = FastAPI(title="TrafficAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router.engine = InferenceEngine()  # type: ignore[attr-defined]
logger.info("GPU enabled: %s", router.engine.device.startswith("cuda"))  # type: ignore[attr-defined]
router.video_jobs = VideoJobStore()  # type: ignore[attr-defined]
app.include_router(router)


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    await handle_stream(websocket, router.engine)  # type: ignore[attr-defined]
