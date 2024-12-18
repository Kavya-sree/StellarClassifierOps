import os
import pandas as pd  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from urllib.parse import urlparse
import mlflow
import numpy as np
import joblib 
from pathlib import Path

from src.StellarClassifier.utils.common import save_json
from src.StellarClassifier.entity.config_entity import ModelEvaluationConfig

#os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/Kavya-sree/StellarClassifierOps.mlflow"
#os.environ["MLFLOW_TRACKING_USERNAME"]="Kavya-sree"
#os.environ["MLFLOW_TRACKING_PASSWORD"]="a51f068524ee98f5c2b17780f7fd183663953438"


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config=config

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, average='weighted')
        recall = recall_score(actual, pred, average='weighted')
        f1 = f1_score(actual, pred, average='weighted')
        return accuracy, precision, recall, f1
    
    def log_into_mlflow(self):

        # Load test data
        test_data = pd.read_csv(self.config.test_data_path)
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column].values

        # Load Label Encoder
        label_encoder = joblib.load(self.config.label_encoder_path)

        # Transform test_y to encoded values
        test_y_encoded = label_encoder.transform(test_y)

        # Load model
        model = joblib.load(self.config.model_path)

    
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

            predicted_qualities = model.predict(test_x)
            (accuracy, precision, recall, f1) = self.eval_metrics(test_y_encoded, predicted_qualities)

            # Saving metrics as local
            scores = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForestModel")

            else: 
                mlflow.sklearn.log_model(model, "model")