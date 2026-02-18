from __future__ import annotations

import io
from pathlib import Path

import requests
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, HttpUrl
from PIL import Image

from src.model_service import model_service
from src.settings import settings
from src.training import TrainingConfig, training_manager

router = APIRouter()

EXPECTED_WIDTH = 360
EXPECTED_HEIGHT = 480


class UrlInferenceRequest(BaseModel):
    image_url: HttpUrl


class FineTuneRequest(BaseModel):
    dataset_dir: str = Field(..., description="Folder with subfolders real/ and spoof/")
    epochs: int = Field(3, ge=1, le=100)
    batch_size: int = Field(8, ge=1, le=256)
    learning_rate: float = Field(1e-4, gt=0)
    output_name: str = Field("finetuned_model.pth")


class LoadModelRequest(BaseModel):
    model_path: str


def _validate_image_size(image: Image.Image) -> None:
    width, height = image.size
    if width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT:
        raise HTTPException(
            status_code=400,
            detail=f"Ukuran wajib {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}px, diterima {width}x{height}px",
        )


@router.post("/predict")
async def predict_image(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File kosong")

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="File bukan gambar valid") from exc

    _validate_image_size(image)

    result = model_service.predict(image)
    return {
        "filename": file.filename,
        "size": {"width": image.width, "height": image.height},
        "prediction": result,
    }


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File kosong")

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="File bukan gambar valid") from exc

    _validate_image_size(image)

    destination = Path(settings.upload_dir) / file.filename
    with destination.open("wb") as out:
        out.write(content)

    return {
        "message": "Upload berhasil",
        "filename": file.filename,
        "path": str(destination),
        "size": {"width": image.width, "height": image.height},
    }


@router.post("/predict/url")
def predict_from_url(req: UrlInferenceRequest) -> dict:
    try:
        response = requests.get(str(req.image_url), timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=400, detail=f"Gagal ambil gambar dari URL: {exc}") from exc

    try:
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Konten URL bukan gambar valid") from exc

    _validate_image_size(image)

    result = model_service.predict(image)
    return {
        "source": str(req.image_url),
        "size": {"width": image.width, "height": image.height},
        "prediction": result,
    }


@router.post("/model/load")
def load_model(req: LoadModelRequest) -> dict:
    model_service.load_weights(req.model_path)
    return {"message": "Model loaded", "model_path": req.model_path}


@router.post("/finetune")
def finetune(req: FineTuneRequest) -> dict:
    dataset_path = Path(req.dataset_dir)
    if not dataset_path.exists():
        raise HTTPException(status_code=400, detail=f"Dataset tidak ditemukan: {dataset_path}")

    job_id = training_manager.create_job(
        TrainingConfig(
            dataset_dir=req.dataset_dir,
            epochs=req.epochs,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            output_name=req.output_name,
        )
    )
    return {"job_id": job_id, "status": "queued"}


@router.get("/finetune/{job_id}")
def finetune_status(job_id: str) -> dict:
    try:
        status = training_manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job tidak ditemukan") from exc
    return {"job_id": job_id, **status}
