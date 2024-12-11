import os
import pandas as pd
from src.StellarClassifier import logger
from src.StellarClassifier.entity.config_entity import (DataValidationConfig)

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def load_data(self):
        """ 
        Load data from CSV
        """
        data = pd.read_csv(self.config.unzip_data_dir)
        logger.info(f"Data loaded from {self.config.unzip_data_dir}")
        return data
    
    def drop_unwanted_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Drops columns that are irrelevant or known to be unnecessary.
        """
        # Define columns to drop
        columns_to_drop = ['obj_ID', 'spec_obj_ID', 'rerun_ID', 'run_ID', 'cam_col', 'field_ID', 'plate', 'MJD', 'fiber_ID', 'alpha', 'delta']
        
        # Drop unwanted columns
        data = data.drop(columns=columns_to_drop, errors='ignore')

        return data


    def validate_all_columns(self, data: pd.DataFrame) -> bool:
        """
        Validates columns in the data against the defined schema"""
        try:
            validation_status = None

            # drop unwanted columns
            data = self.drop_unwanted_columns(data)

            # validate against schema
            all_cols = list(data.columns)
            all_schema = self.config.all_schema.keys()

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f: 
                        f.write(f"Validation status: {validation_status}")

                else: 
                    validation_status = True 
                    with open(self.config.STATUS_FILE, 'w') as f: 
                        f.write(f"Validation status: {validation_status}")
        
            return validation_status
        
        except Exception as e:
            raise e

    

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the data by handling missing values, removing outliers, and correcting data types.
        """
        # Replace placeholder value with NaN for consistency
        data.replace(-9999.0, pd.NA, inplace=True)

        # Drop rows where any column has NaN values
        data.dropna(inplace=True)

        return data
    
    def save_cleaned_data(self, data: pd.DataFrame):
        """
        Saves the cleaned data to a CSV file.
        """
        try:
            data.to_csv(self.config.cleaned_data_dir, index=False)
            logger.info(f"Cleaned data saved to {self.config.cleaned_data_dir}")
        except Exception as e:
            logger.error(f"Error saving cleaned data: {e}")
            raise