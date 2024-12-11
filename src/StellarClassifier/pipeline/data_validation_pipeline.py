from src.StellarClassifier.config.configuration import ConfigurationManager
from src.StellarClassifier.components.data_validation import DataValidation
from src.StellarClassifier import logger

STAGE_NAME = "Data Validation stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_validation(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation =  DataValidation(config=data_validation_config)

        data = data_validation.load_data()
        data = data_validation.drop_unwanted_columns(data)

        # validate columns
        validation_status= data_validation.validate_all_columns(data)
        if not validation_status:
            logger.error("Data validation failed. Aborting pipeline.")
            return
        
        # clean data
        cleaned_data = data_validation.clean_data(data)

        # save cleaned data
        data_validation.save_cleaned_data(cleaned_data)


if __name__=='__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = DataValidationTrainingPipeline()
        obj.initiate_data_validation()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

