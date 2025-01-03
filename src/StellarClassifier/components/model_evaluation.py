import os
from dotenv import load_dotenv, find_dotenv

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from urllib.parse import urlparse
from src.StellarClassifier.utils.common import save_json
import tempfile


import mlflow
import joblib
from pathlib import Path
from src.StellarClassifier.entity.config_entity import ModelEvaluationConfig

# Locate and load .env file
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        """Initializes the ModelEvaluation class with the configuration."""
        self.config = config

    def eval_metrics(self, actual, pred):
        """Evaluate model metrics including accuracy, precision, recall, and f1 score."""
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, average='weighted')
        recall = recall_score(actual, pred, average='weighted')
        f1 = f1_score(actual, pred, average='weighted')
        return accuracy, precision, recall, f1

    def log_into_mlflow(self, model_type: str):
        """Log model evaluation metrics into MLflow."""
        try:
            # Load test data
            test_data = pd.read_csv(self.config.test_data_path)
            test_x = test_data.drop([self.config.target_column], axis=1)
            test_y = test_data[self.config.target_column].values

            # Load Label Encoder
            label_encoder = joblib.load(self.config.label_encoder_path)

            # Transform test_y to encoded values
            test_y_encoded = label_encoder.transform(test_y)

            # Load model from the saved path
            model_path = self.config.model_path.format(model_type=model_type)
            model = joblib.load(model_path)

            # Set up MLflow tracking URI
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Define a meaningful run name
            run_name = f"Eval_{model_type}_max_evals_{self.config.max_evals}_cv_{self.config.cv}"

            with mlflow.start_run(run_name=run_name):
                # Predict the target values
                predicted_qualities = model.predict(test_x)
                (accuracy, precision, recall, f1) = self.eval_metrics(test_y_encoded, predicted_qualities)

                # Save metrics locally in a JSON file
                scores = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
                save_json(path=Path(self.config.metric_file_template.format(model_type=model_type)), data=scores)


                # Log metrics to MLflow
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)

                # Compute the confusion matrix
                cm = confusion_matrix(test_y_encoded, predicted_qualities)

                # Save confusion matrix as an image
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')

                # Save the plot to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                    plt.savefig(tmpfile.name)
                    plt.close()
                    # Log confusion matrix image to MLflow
                    mlflow.log_artifact(tmpfile.name, "confusion_matrix")

                # Extract hyperparameters used during the run
                used_params = {}
                if hasattr(model, "get_params"):
                    used_params = model.get_params()

                # Log model parameters to MLflow
                mlflow.log_params(used_params)

                # Log the model to MLFlow registry
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name=model_type)
                else:
                    mlflow.sklearn.log_model(model, "model")

                print(f"Model evaluation metrics logged for {model_type}")

        except Exception as e:
            print(f"Error during model evaluation: {str(e)}")
            raise e
