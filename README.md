# Face Anti-Spoofing Retraining API

Service API untuk mendeteksi wajah asli (live) atau palsu (spoof) menggunakan model Deep Learning. Didukung dengan fitur Fine-tuning untuk melakukan training ulang model dengan dataset custom.

## Daftar Isi

1. [Requirements](#requirements)
2. [Struktur Project](#struktur-project)
3. [Cara Install](#cara-install)
4. [Cara Menjalankan Docker](#cara-menjalankan-docker)
5. [Cara Menghentikan dan Menghapus Docker](#cara-menghentikan-dan-menghapus-docker)
6. [Pemanfaatan API](#pemanfaatan-api)
   - [Health Check](#1-health-check)
   - [Prediksi dari File Gambar](#2-prediksi-dari-file-gambar)
   - [Prediksi dari URL](#3-prediksi-dari-url)
   - [Fine-tuning Model](#4-fine-tuning-model)
   - [Load Model Baru](#5-load-model-baru)
7. [Format Dataset untuk Fine-tuning](#format-dataset-untuk-fine-tuning)
8. [Troubleshooting](#troubleshooting)
9. [Informasi Model](#informasi-model)

---

## Requirements

- Docker & Docker Compose
- GPU (opsional, untuk akselerasi)
- Akun Docker dengan GPU support (jika menggunakan NVIDIA GPU)

---

## Struktur Project

```
face-antiSpoofing-retraining/
├── app.py                  # Entry point aplikasi
├── Dockerfile              # Konfigurasi Docker
├── docker-compose.yml      # Konfigurasi Docker Compose
├── requirements.txt        # Dependencies Python
├── src/
│   ├── main.py            # Konfigurasi FastAPI
│   ├── routes.py          # Definisi endpoint API
│   ├── model_service.py   # Logika loading model & prediksi
│   ├── settings.py        # Konfigurasi aplikasi
│   ├── training.py        # Logika fine-tuning
│   └── NN.py              # Definisi arsitektur model
├── data/
│   ├── models/            # Folder untuk menyimpan model .pth
│   │   └── anti_spoofing.pth
│   └── uploads/          # Folder untuk upload gambar
├── scripts/
│   └── download_model.sh # Script download model
└── README.md
```

---

## Cara Install

### 1. Clone repository (jika belum)
```bash
git clone <repo-url>
cd face-antiSpoofing-retraining
```

### 2. Buat folder yang diperlukan
```bash
mkdir -p data/models data/uploads
```

### 3. Download model (opsional)

Download model dari GitHub hairymax/Face-AntiSpoofing:
- Link: https://github.com/hairymax/Face-AntiSpoofing/tree/main/saved_models
- Model yang disarankan: `AntiSpoofing_bin_1.5_128.pth`
- Simpan di `data/models/anti_spoofing.pth`

Atau gunakan script:
```bash
bash scripts/download_model.sh
```

### 4. Konfigurasi environment (opsional)
```bash
cp .env.example .env
```

Edit file `.env` jika perlu:
```env
APP_HOST=0.0.0.0
APP_PORT=8000
MODEL_INPUT_SIZE=224
DEFAULT_MODEL_PATH=data/models/anti_spoofing.pth
```

---

## Cara Menjalankan Docker

### Menjalankan container (build + up)
```bash
sudo docker compose down
sudo docker compose build --no-cache
sudo docker compose up -d
```

### Menjalankan container (tanpa build ulang)
```bash
sudo docker compose up -d
```

### Melihat logs
```bash
sudo docker compose logs -f anti-spoof-api
```

### Melihat status container
```bash
sudo docker compose ps
```

### API akan aktif di:
```
http://localhost:8000
```

### Dokumentasi Swagger:
```
http://localhost:8000/docs
```

---

## Cara Menghentikan dan Menghapus Docker

### Menghentikan container (tidak dihapus)
```bash
sudo docker compose down
```

### Menghapus container dan network
```bash
sudo docker compose down --volumes
```

### Menghapus container, network, dan image
```bash
sudo docker compose down --rmi all
```

### Menghapus semua yang terkait (container, image, volume, cache)
```bash
sudo docker compose down --rmi all --volumes
sudo docker system prune -a
```

---

## Pemanfaatan API

### 1. Health Check

Cek status service apakah sedang berjalan.

**Endpoint:** `GET /health`

**Contoh cURL:**
```bash
curl -X GET "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda"
}
```

---

### 2. Prediksi dari File Gambar

Mendeteksi wajah asli (live) atau palsu (spoof) dari file gambar.

**Endpoint:** `POST /predict`

**Parameters:**
- `file`: File gambar (multipart/form-data)

**Contoh cURL:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@data/uploads/foto_wajah.jpg"
```

**Response:**
```json
{
  "filename": "foto_wajah.jpg",
  "size": {
    "width": 360,
    "height": 480
  },
  "prediction": {
    "label": "live",
    "live_score": 0.999998,
    "spoof_score": 0.000002,
    "threshold": 0.5,
    "device": "cuda"
  }
}
```

**Penjelasan:**
- `label`: "live" (wajah asli) atau "spoof" (wajah palsu/fake)
- `live_score`: Probabilitas wajah asli (0-1)
- `spoof_score`: Probabilitas wajah palsu (0-1)
- `threshold`: Batas ambang untuk menentukan label
- `device`: Device yang digunakan ("cuda" = GPU, "cpu" = CPU)

---

### 3. Prediksi dari URL

Mendeteksi wajah dari URL gambar.

**Endpoint:** `POST /predict/url`

**Body (JSON):**
```json
{
  "image_url": "https://example.com/face.jpg"
}
```

**Contoh cURL:**
```bash
curl -X POST "http://localhost:8000/predict/url" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/face.jpg"}'
```

**Response:**
```json
{
  "prediction": {
    "label": "live",
    "live_score": 0.95,
    "spoof_score": 0.05,
    "threshold": 0.5,
    "device": "cuda"
  },
  "image_source": "url"
}
```

---

### 4. Fine-tuning Model

Melakukan training ulang model dengan dataset custom.

**Endpoint:** `POST /finetune`

**Body (JSON):**
```json
{
  "dataset_dir": "data/datasets/mydataset",
  "epochs": 3,
  "batch_size": 8,
  "learning_rate": 0.0001,
  "output_name": "my_finetuned_model.pth"
}
```

**Parameter:**
- `dataset_dir` (wajib): Path ke folder dataset
- `epochs` (opsional, default: 3): Jumlah epoch training
- `batch_size` (opsional, default: 8): Batch size
- `learning_rate` (opsional, default: 0.0001): Learning rate
- `output_name` (opsional): Nama file model hasil training

**Contoh cURL:**
```bash
curl -X POST "http://localhost:8000/finetune" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_dir": "data/datasets/mydataset",
    "epochs": 5,
    "batch_size": 16
  }'
```

**Response:**
```json
{
  "job_id": "ft_1234567890",
  "status": "started",
  "message": "Fine-tuning started"
}
```

### Cek Status Fine-tuning

**Endpoint:** `GET /finetune/{job_id}`

**Contoh cURL:**
```bash
curl -X GET "http://localhost:8000/finetune/ft_1234567890"
```

**Response:**
```json
{
  "job_id": "ft_1234567890",
  "status": "completed",
  "output_path": "data/models/my_finetuned_model.pth",
  "metrics": {
    "train_loss": 0.123,
    "val_accuracy": 0.95
  }
}
```

---

### 5. Load Model Baru

Memuat model .pth baru tanpa restart service.

**Endpoint:** `POST /model/load`

**Body (JSON):**
```json
{
  "model_path": "data/models/my_finetuned_model.pth"
}
```

**Contoh cURL:**
```bash
curl -X POST "http://localhost:8000/model/load" \
  -H "Content-Type: application/json" \
  -d '{"model_path": "data/models/my_finetuned_model.pth"}'
```

**Response:**
```json
{
  "status": "success",
  "message": "Model loaded successfully",
  "model_path": "data/models/my_finetuned_model.pth"
}
```

---

## Format Dataset untuk Fine-tuning

Siapkan dataset dengan struktur folder sebagai berikut:

```
data/datasets/
└── mydataset/
    ├── real/           # Gambar wajah asli (live)
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    └── spoof/          # Gambar wajah palsu (spoof)
        ├── img1.jpg
        ├── img2.jpg
        └── ...
```

**Catatan:**
- `real` = wajah asli (label: 1)
- `spoof` = wajah palsu seperti foto yang dicetak, foto di layar HP, dll (label: 0)
- Support format: .jpg, .jpeg, .png
- Disarankan menggunakan gambar dengan wajah centered

---

## Troubleshooting

### 1. Container tidak bisa start

Cek logs:
```bash
sudo docker compose logs -f anti-spoof-api
```

### 2. Model tidak terdeteksi

Pastikan file model ada di `data/models/`:
```bash
ls -la data/models/
```

### 3. GPU tidak terdeteksi

Pastikan NVIDIA Docker terinstall:
```bash
nvidia-smi
sudo docker compose build --no-cache
sudo docker compose up -d
```

### 4. CUDA out of memory

Kurangi batch_size di fine-tuning atau gunakan CPU:
```bash
# Edit settings.py, ubah:
device = "cpu"
```

### 5. Port sudah digunakan

Ganti port di `.env`:
```env
APP_PORT=8001
```

---

## Informasi Model

### Model yang Digunakan

Repository: [hairymax/Face-Around Spoofing](https://github.com/hairymax/Face-AntiSpoofing)

Model yang disarankan:
- `AntiSpoofing_bin_1.5_128.pth` (size: ~2.9MB)

### Arsitektur Model

Model menggunakan arsitektur **MiniFASNetV2SE** dengan:
- Input size: 128x128 atau 224x224 (sesuai konfigurasi)
- Output: 2 kelas (live/spoof)

### Threshold

- Default threshold: 0.55
- Jika `spoof_score >= 0.55` → label = "spoof"
- Jika `spoof_score < 0.55` → label = "live"

Threshold bisa diubah di `src/model_service.py`:

```python
threshold = 0.55  # Ubah nilai ini
```

---

## Contoh Penggunaan Lengkap

### 1. Jalankan service
```bash
sudo docker compose up -d
```

### 2. Cek status
```bash
curl -X GET "http://localhost:8000/health"
```

### 3. Prediksi gambar
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@data/uploads/wajah.jpg"
```

### 4. Fine-tuning dengan dataset sendiri
```bash
# Pastikan dataset sudah siap di data/datasets/mydataset/
curl -X POST "http://localhost:8000/finetune" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_dir": "data/datasets/mydataset",
    "epochs": 10
  }'
```

### 5. Load model hasil fine-tuning
```bash
curl -X POST "http://localhost:8000/model/load" \
  -H "Content-Type: application/json" \
  -d '{"model_path": "data/models/mydataset_model.pth"}'
```

---

## Lisensi

MIT License

