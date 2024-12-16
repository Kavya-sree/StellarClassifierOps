import pandas as pd  
import os
from src.StellarClassifier import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

from src.StellarClassifier.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

        # Encode target column
        le = LabelEncoder()
        train_y = le.fit_transform(train_y)
        test_y = le.transform(test_y)

        # Save the label encoder for use during evaluation
        label_encoder_path = os.path.join(self.config.root_dir, "label_encoder.pkl")
        joblib.dump(le, label_encoder_path)

        classifier = RandomForestClassifier(n_estimators=self.config.n_estimators, random_state=42)
        classifier.fit(train_x, train_y)

        joblib.dump(classifier, os.path.join(self.config.root_dir, self.config.model_name))