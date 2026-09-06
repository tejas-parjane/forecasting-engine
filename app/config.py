"""Configuration for the Forecasting Engine application."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


@dataclass
class Settings:
    app_name: str = os.getenv("APP_NAME", "Forecasting Engine")
    app_version: str = os.getenv("APP_VERSION", "1.0.0")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    sample_data_file: Path = field(
        default_factory=lambda: Path(os.getenv("SAMPLE_DATA_FILE", "data/sample/business_demand.csv"))
    )
    sample_data_seed: int = int(os.getenv("SAMPLE_DATA_SEED", "42"))

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    ai_enabled_flag: bool = os.getenv("AI_ENABLED", "false").lower() == "true"
    ai_provider: str = os.getenv("AI_PROVIDER", "openai")
    ai_model: str = os.getenv("AI_MODEL", "gpt-4o-mini")
    ai_temperature: float = float(os.getenv("AI_TEMPERATURE", "0.2"))
    ai_max_tokens: int = int(os.getenv("AI_MAX_TOKENS", "800"))

    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    @property
    def ai_enabled(self) -> bool:
        """The LLM layer is OFF by default and requires BOTH an explicit
        AI_ENABLED=true flag and an OPENAI_API_KEY. Without it the engine
        uses the deterministic rule-based decision briefs."""
        return self.ai_enabled_flag and bool(self.openai_api_key)

    def is_sample_data_abs(self) -> bool:
        return Path(self.sample_data_file).is_absolute()

    def resolve_sample_data(self) -> Path:
        if self.is_sample_data_abs():
            return Path(self.sample_data_file)
        return (ROOT_DIR / self.sample_data_file).resolve()


settings = Settings()
