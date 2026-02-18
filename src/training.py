from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Thread

import torch
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.model_service import model_service
from src.settings import settings


@dataclass
class TrainingConfig:
    dataset_dir: str
    epochs: int
    batch_size: int
    learning_rate: float
    output_name: str


class BinarySpoofDataset(Dataset):
    def __init__(self, dataset_dir: str, image_size: int) -> None:
        self.samples: list[tuple[Path, float]] = []
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        root = Path(dataset_dir)
        mapping = {"spoof": 0.0, "real": 1.0}

        for folder_name, label in mapping.items():
            folder = root / folder_name
            if not folder.exists():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                for img_path in folder.rglob(ext):
                    self.samples.append((img_path, label))

        if not self.samples:
            raise ValueError(
                "Dataset kosong. Gunakan struktur: dataset_dir/real/*.jpg dan dataset_dir/spoof/*.jpg"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, torch.tensor([label], dtype=torch.float32)


class TrainingManager:
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self.lock = Lock()

    def create_job(self, cfg: TrainingConfig) -> str:
        job_id = str(uuid.uuid4())
        with self.lock:
            self.jobs[job_id] = {
                "status": "queued",
                "message": "Job created",
                "output_model": None,
            }

        thread = Thread(target=self._run, args=(job_id, cfg), daemon=True)
        thread.start()
        return job_id

    def get(self, job_id: str) -> dict:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError("Job not found")
            return self.jobs[job_id]

    def _set(self, job_id: str, **kwargs: object) -> None:
        with self.lock:
            self.jobs[job_id].update(kwargs)

    def _run(self, job_id: str, cfg: TrainingConfig) -> None:
        try:
            self._set(job_id, status="running", message="Preparing dataset")
            dataset = BinarySpoofDataset(cfg.dataset_dir, settings.model_input_size)
            loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)

            model = model_service.model
            model.train()
            criterion = nn.BCEWithLogitsLoss()
            optimizer = Adam(model.parameters(), lr=cfg.learning_rate)

            device = model_service.device
            self._set(job_id, message=f"Training on {device} with {len(dataset)} samples")

            for epoch in range(cfg.epochs):
                running_loss = 0.0
                for images, labels in loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    logits = model(images)
                    if logits.ndim == 1:
                        logits = logits.unsqueeze(1)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                avg_loss = running_loss / max(1, len(loader))
                self._set(job_id, message=f"Epoch {epoch + 1}/{cfg.epochs} loss={avg_loss:.6f}")

            output_path = Path(settings.models_dir) / cfg.output_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)

            self._set(
                job_id,
                status="completed",
                message="Training completed",
                output_model=str(output_path),
            )
            model.eval()

        except Exception as exc:  # noqa: BLE001
            self._set(job_id, status="failed", message=str(exc))


training_manager = TrainingManager()
