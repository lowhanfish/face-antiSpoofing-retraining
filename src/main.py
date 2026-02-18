from pathlib import Path

from fastapi import FastAPI

from src.routes import router
from src.settings import settings


def ensure_directories() -> None:
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.models_dir).mkdir(parents=True, exist_ok=True)


app = FastAPI(
    title="Face Anti-Spoofing API",
    version="0.1.0",
    description="FastAPI service for anti-spoofing inference and fine-tuning with .pth models.",
)


@app.on_event("startup")
def startup_event() -> None:
    ensure_directories()


app.include_router(router)
