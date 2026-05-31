from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StrideSettings(BaseSettings):
    """Machine-local settings loaded from environment or .env.local."""

    data_root: Path = Field(default=Path("stride-data"), alias="STRIDE_DATA_ROOT")
    models_root: Path = Field(default=Path("models"), alias="STRIDE_MODELS_ROOT")
    wandb_project: str = Field(default="stride", alias="STRIDE_WANDB_PROJECT")
    wandb_entity: str | None = Field(default=None, alias="STRIDE_WANDB_ENTITY")
    wandb_api_key: str | None = Field(default=None, alias="STRIDE_WANDB_API_KEY")
    log_level: str = Field(default="INFO", alias="STRIDE_LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )
