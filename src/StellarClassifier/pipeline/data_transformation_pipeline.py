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
            with open(Path("artifacts/data_validation/status.txt"), 'r') as f:
                status = f.read().split(" ")[-1]
            if status == "True":
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)

                # Train-test split
                train, test = data_transformation.train_test_spliting()

                # Apply SMOTE
                resampled_train = data_transformation.apply_smote(train)

                # Apply Magnitude Difference Transformation (u_g, g_r, r_i, i_z)
                transformed_train = data_transformation.compute_magnitude_diff(resampled_train)
                transformed_test = data_transformation.compute_magnitude_diff(test)

                # Normalize data
                normalized_train = data_transformation.normalize_data(transformed_train)
                normalized_test = data_transformation.normalize_data(transformed_test)

                # Save processed data
                normalized_train.to_csv(os.path.join(config.get_data_transformation_config().root_dir, "normalized_train.csv"), index=False)
                normalized_test.to_csv(os.path.join(config.get_data_transformation_config().root_dir, "normalized_test.csv"), index=False)

                logger.info("Data Transformation pipeline completed, including Magnitude Difference Transformation.")

            else:
                raise Exception("Your scheme is not valid")
            
        except Exception as e:
            logger.error(f"Error during data transformation pipeline: {e}")
            print(e)