import yaml
import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ClipConfig:
    min: float = 0.5
    max: float = 30.0
    default: float = 5.0
    overlap: float = 0.5


@dataclass
class ProcessingConfig:
    batch_size: int = 32
    max_parallel: int = 100
    cache_ttl: int = 3600
    device: str = "cuda"
    precision: str = "fp16"


@dataclass
class ModelConfig:
    video_encoder: str = "openai/clip-vit-base-patch32"
    text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    blip2_model: str = "Salesforce/blip2-opt-2.7b"
    device: str = "cpu"
    optimization: str = "none"


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout: int = 300
    max_file_size: int = 500000000


@dataclass
class StorageConfig:
    temp_dir: str = "/tmp/video_contextualizer"
    cache_dir: str = "/tmp/video_contextualizer/cache"
    max_storage_mb: int = 10000


@dataclass
class PerformanceConfig:
    target_accuracy: float = 0.85
    max_processing_time_ms: int = 500
    gpu_utilization_target: float = 0.8
    memory_limit_gb: int = 16


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/contextualizer.log"


@dataclass
class SystemConfig:
    clip_duration: ClipConfig
    processing: ProcessingConfig
    models: ModelConfig
    api: APIConfig
    storage: StorageConfig
    performance: PerformanceConfig
    logging: LoggingConfig

    @classmethod
    def from_yaml(cls, config_path: str) -> "SystemConfig":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return cls(
            clip_duration=ClipConfig(**config_data.get('clip_duration', {})),
            processing=ProcessingConfig(**config_data.get('processing', {})),
            models=ModelConfig(**config_data.get('models', {})),
            api=APIConfig(**config_data.get('api', {})),
            storage=StorageConfig(**config_data.get('storage', {})),
            performance=PerformanceConfig(**config_data.get('performance', {})),
            logging=LoggingConfig(**config_data.get('logging', {}))
        )

    @classmethod
    def load_default(cls) -> "SystemConfig":
        """Load default configuration."""
        # Return hardcoded default configuration instead of loading from file
        return cls(
            clip_duration=ClipConfig(),
            processing=ProcessingConfig(device="cpu", precision="fp32"),
            models=ModelConfig(device="cpu"),
            api=APIConfig(),
            storage=StorageConfig(),
            performance=PerformanceConfig(),
            logging=LoggingConfig()
        )


def get_config() -> SystemConfig:
    """Get system configuration from environment or default."""
    config_path = os.getenv("CONFIG_PATH")
    if config_path and os.path.exists(config_path):
        return SystemConfig.from_yaml(config_path)
    return SystemConfig.load_default()