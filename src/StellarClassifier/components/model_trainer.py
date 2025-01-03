from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

from xgboost import XGBClassifier
import pandas as pd
from hyperopt import fmin, tpe, Trials, STATUS_OK, STATUS_FAIL, hp
import os
from sklearn.preprocessing import LabelEncoder
import joblib
import mlflow
import numpy as np
from sklearn.model_selection import cross_val_score
from src.StellarClassifier.entity.config_entity import ModelTrainerConfig
from sklearn.model_selection import StratifiedKFold


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def load_data(self):
        """Load train and test datasets."""
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop(columns=[self.config.target_column])
        test_x = test_data.drop(columns=[self.config.target_column])
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

        # Encode target column
        le = LabelEncoder()
        train_y = le.fit_transform(train_y)
        test_y = le.transform(test_y)

        # Save the label encoder
        label_encoder_path = self.config.label_encoder_path
        if not os.path.exists(os.path.dirname(label_encoder_path)):
            os.makedirs(os.path.dirname(label_encoder_path))
        joblib.dump(le, label_encoder_path)

        return train_x, test_x, train_y, test_y

    def parse_param_space(self, param_space):
        """Parse the param_space from YAML configuration into Hyperopt's space format."""
        parsed_space = {}
        for param, details in param_space.items():
            param_type = details.get('type')
            if param_type == 'quniform':
                parsed_space[param] = hp.quniform(param, details['low'], details['high'], details['q'])
            elif param_type == 'uniform':
                parsed_space[param] = hp.uniform(param, details['low'], details['high'])
            elif param_type == 'choice':
                parsed_space[param] = hp.choice(param, details['values'])
            else:
                raise ValueError(f"Unsupported type {param_type} for {param}")
        return parsed_space

    def get_search_space(self):
        """Retrieve the hyperparameter search space from config."""
        param_space = self.config.param_space
        if not param_space:
            raise ValueError(f"No parameter space defined for model type: {self.config.model_type}")
        return self.parse_param_space(param_space)

    def train_model(self):
        """Train the model using hyperparameter optimization with cross-validation."""
        train_x, test_x, train_y, test_y = self.load_data()
        model_type = self.config.model_type
        cv = StratifiedKFold(n_splits=self.config.cv)

        max_evals = self.config.max_evals

        if model_type not in ["RandomForest", "XGBoost", "LightGBM"]:
            raise ValueError(f"Unsupported model type: {model_type}")

        search_space = self.get_search_space()

        # MLflow run name
        run_name = f"Train_{model_type}_maxEvals{max_evals}_cv{cv.n_splits}"

        def objective(params):
            """Objective function for hyperparameter optimization."""
            try:
                model = self.initialize_model(model_type, params)

                # Perform cross-validation
                cv_scores = cross_val_score(model, train_x, train_y, cv=cv, n_jobs=-1)
                avg_cv_score = np.mean(cv_scores)

                return {"loss": -avg_cv_score, "status": STATUS_OK}
            except Exception as e:
                print(f"Error during model training: {e}")
                return {"loss": float('inf'), "status": STATUS_FAIL}  # Return high loss in case of error

        # Optimize hyperparameters
        trials = Trials()

        try:
            with mlflow.start_run(run_name=run_name):
                best_params = fmin(
                    fn=objective,
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    rstate=np.random.default_rng(self.config.random_state),
                    trials=trials,
                )

                # Finalize the model
                best_model = self.initialize_model(model_type, best_params)
                best_model.fit(train_x, train_y)

                # Save the model
                model_path = self.config.model_file_template.format(model_type=model_type)
                joblib.dump(best_model, model_path)

                # Compute metrics
                train_accuracy = best_model.score(train_x, train_y)
                test_accuracy = best_model.score(test_x, test_y)
                metrics = {"train_accuracy": train_accuracy, "test_accuracy": test_accuracy}

                # Log metrics to MLflow (no local saving)
                mlflow.log_metrics(metrics)

                print(f"Best hyperparameters for {model_type}: {best_params}")
                print(f"Model saved at {model_path}")

                # Log model parameters and metrics to MLflow
                mlflow.log_params(best_params)
                mlflow.sklearn.log_model(best_model, "model")

        except Exception as e:
            print(f"Error during model training: {e}")
            raise

    def initialize_model(self, model_type, params):
        """Initialize the model with the given parameters."""
        if model_type == "RandomForest":
            return RandomForestClassifier(
                n_estimators=int(params["n_estimators"]),  
                max_depth=int(params["max_depth"]),
                min_samples_split=int(params["min_samples_split"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                random_state=self.config.random_state,
                n_jobs=-1
            )
        elif model_type == "XGBoost":
            return XGBClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                learning_rate=float(params["learning_rate"]),
                random_state=self.config.random_state,
                n_jobs=-1
            )
        elif model_type == "LightGBM":
            # return lgb.LGBMClassifier(random_state=self.config.random_state, n_jobs=-1)
            return lgb.LGBMClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                learning_rate=float(params["learning_rate"]),
                subsample=float(params["subsample"]),
                num_leaves=int(params["num_leaves"]),  
                min_data_in_leaf=int(params["min_data_in_leaf"]),  # Controls overfitting
                feature_fraction=float(params["feature_fraction"]),  # Fraction of features used to build trees
                random_state=self.config.random_state,
                n_jobs=-1
    )

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

