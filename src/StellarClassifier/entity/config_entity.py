from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path 
    unzip_dir: Path

@dataclass
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict
    cleaned_data_dir: Path

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    test_size: float
    random_state: int
    target_column: str
    numerical_columns: list