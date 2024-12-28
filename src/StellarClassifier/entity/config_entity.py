
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union, List  
from hyperopt import hp

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

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_file_template: str  
    label_encoder_path: str
    model_type: str  
    model_selection: str
    hyperparameter_tuning: bool
    param_space: Dict[str, Union[hp.choice, hp.uniform, hp.quniform, List[Union[str, float, int]]]]  # Corrected usage
    cv: int  
    random_state: int
    mlflow_experiment_name: str
    target_column: str
    max_evals: int

@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    label_encoder_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str
