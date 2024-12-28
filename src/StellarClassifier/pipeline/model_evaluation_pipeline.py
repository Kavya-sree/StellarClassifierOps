from src.StellarClassifier.config.configuration import ConfigurationManager
from src.StellarClassifier.components.model_evaluation import ModelEvaluation
from src.StellarClassifier import logger

STAGE_NAME = "Model Evaluation stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def initiate_model_evaluation(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()

        # Extract model_type from the configuration
        model_type = config.params['model_selection']['model_type']

        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.log_into_mlflow(model_type)