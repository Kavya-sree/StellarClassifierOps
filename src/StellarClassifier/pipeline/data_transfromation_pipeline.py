from src.StellarClassifier.config.configuration import ConfigurationManager
from src.StellarClassifier.components.data_transformation import DataTransformation
from src.StellarClassifier import logger

import os
from pathlib import Path

STAGE_NAME="Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):

        try:
            with open(Path("artifacts/data_validation/status.txt"),'r') as f: 
                status=f.read().split(" ")[-1]
            if status=="True":
                config = ConfigurationManager()
                data_trasformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_trasformation_config)

                # Train-test split
                train, test = data_transformation.train_test_spliting()

                # Apply SMOTE
                resampled_train = data_transformation.apply_smote(train)

                # Normalize data
                normalized_train = data_transformation.normalize_data(resampled_train)
                normalized_test = data_transformation.normalize_data(test)

                # Save processed data
                normalized_train.to_csv(os.path.join(config.get_data_transformation_config().root_dir, "normalized_train.csv"), index=False)
                normalized_test.to_csv(os.path.join(config.get_data_transformation_config().root_dir, "normalized_test.csv"), index=False)

                logger.info("Data Transformation pipeline completed.")

            else:  
                raise Exception("Your scheme is not valid")
            
        except Exception as e: 
            print(e)