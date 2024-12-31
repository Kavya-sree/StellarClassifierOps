import os
from src.StellarClassifier import logger
from src.StellarClassifier.entity.config_entity import DataTransformationConfig
from sklearn.model_selection import train_test_split
import pandas as pd


from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from imblearn.over_sampling import SMOTE

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    def train_test_spliting(self):
        # Load data
        data = pd.read_csv(self.config.data_path)
        logger.info("Loaded data for train-test split.")

        # Split data
        train, test = train_test_split(
            data,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        logger.info("Split data into training and test sets.")
        return train, test

    def apply_smote(self, train_data):
        logger.info("Applying SMOTE for class imbalance...")
        smote = SMOTE(random_state=self.config.random_state)
        X = train_data.drop(columns=[self.config.target_column])
        y = train_data[self.config.target_column]

        #shape before SMOTE
        logger.info(f"shape before SMOTE: Features: {X.shape}, Target: {y.shape}")

        X_resampled, y_resampled = smote.fit_resample(X, y)

        logger.info(f"Shape after SMOTE: Features: {X_resampled.shape}, Target: {y_resampled.shape}")

        resampled_data = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled)], axis=1)
        logger.info("SMOTE applied successfully.")
        return resampled_data

    
    def compute_magnitude_diff(self, data):
        logger.info("Computing magnitude differences...")

        # Compute the magnitude differences for the required pairs
        data['u_g'] = data['u'] - data['g']
        data['g_r'] = data['g'] - data['r']
        data['r_i'] = data['r'] - data['i']
        data['i_z'] = data['i'] - data['z']

        logger.info("Magnitude differences computed.")
        return data
    
    def normalize_data(self, data):
        logger.info("Normalizing data...")
        scaler = MinMaxScaler()

        # Normalize only numerical columns
        numerical_data = data[self.config.numerical_columns]
        scaled_values = scaler.fit_transform(numerical_data)
        
        # Replace original columns with normalized values
        data[self.config.numerical_columns] = scaled_values
        logger.info("Data normalization completed.")
        return data
