from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    upload_dir: str = "data/uploads"
    models_dir: str = "data/models"

    # If set, it should be a dotted path to model class, e.g.
    # "Face-AntiSpoofing.src.models.FaceAntiSpoofingModel"
    model_class_path: str = ""

    # If set, load this .pth at startup.
    default_model_path: str = "data/models/AntiSpoofing_bin_1.5_128.pth"
    # Sesuai dengan preprocessing ONNX dari hairymax
    model_input_size: int = 128

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
