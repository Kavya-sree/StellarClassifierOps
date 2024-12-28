from src.StellarClassifier.constants import *
from src.StellarClassifier.utils.common import read_yaml, create_directories

from src.StellarClassifier.entity.config_entity import (DataIngestionConfig, DataValidationConfig, DataTransformationConfig, 
                                                        ModelTrainerConfig, ModelEvaluationConfig)

class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH,
                 schema_filepath=SCHEMA_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        # Create directories defined in the config if they don't exist
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        create_directories([config.root_dir])

        return DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir=config.unzip_data_dir,
            all_schema=schema,
            cleaned_data_dir=config.cleaned_data_dir
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params = self.params.data_transformation
        schema = self.schema
        create_directories([config.root_dir])

        return DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            test_size=params.test_size,
            random_state=params.random_state,
            target_column=schema.TARGET_COLUMN['name'],
            numerical_columns=[col for col, dtype in schema['COLUMNS'].items() if dtype == 'float64']
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        hyperopt_params = self.params.get('hyperopt', {})

        create_directories([config.root_dir])

        model_type = self.params['model_selection']['model_type']
        cv_folds = self.params['model_selection']['cv']
        random_state = self.params['model_selection']['random_state']
        mlflow_experiment_name = self.params['model_selection']['mlflow_experiment_name']
        hyperparameter_tuning = self.params['model_selection']['hyperparameter_tuning']

        hyperopt_params = self.params.get('hyperopt', {})
        max_evals = hyperopt_params.get('max_evals', {})

        # Fetch model selection and target column
        model_selection = {
            'model_type': model_type,
            'cv': cv_folds,
            'random_state': random_state,
            'mlflow_experiment_name': mlflow_experiment_name,
            'hyperparameter_tuning': hyperparameter_tuning
        }
        target_column = self.schema.TARGET_COLUMN['name'] 

        param_space = self.params.get(model_type, {}).get('param_space', {})

        label_encoder_path = config.get('label_encoder_path')

        # Print configuration details for clarity
        print(f"Model type from params.yaml: {model_type}")
        print(f"Cross-validation folds: {cv_folds}")
        print(f"Hyperparameter tuning enabled: {hyperparameter_tuning}")

        return ModelTrainerConfig(
            root_dir=self.config.artifacts_root,
            model_type=model_type,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_file_template=config.model_file_template,
            label_encoder_path=label_encoder_path,
            cv=cv_folds,
            random_state=random_state,
            mlflow_experiment_name=mlflow_experiment_name,
            hyperparameter_tuning=hyperparameter_tuning,  
            max_evals=max_evals,
            param_space=param_space,
            model_selection=model_selection,  
            target_column=target_column
            )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params
        schema = self.schema.TARGET_COLUMN

        model_type = self.params.get('model_selection', {}).get('model_type', None)

        create_directories([config.root_dir])

        return ModelEvaluationConfig(
        root_dir=config.root_dir,
        test_data_path=config.test_data_path,
        model_path=config.model_path.format(model_type=model_type),  # Dynamic model path based on model type
        label_encoder_path=config.label_encoder_path,
        all_params= params,
        metric_file_name=config.metric_file_name,
        target_column=schema.name,
        mlflow_uri="https://dagshub.com/Kavya-sree/StellarClassifierOps.mlflow"
        )