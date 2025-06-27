import json
import sys
import os

import pandas as pd
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging

from src.utils.main_utils import read_yaml_file , write_yaml_file 
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataValidationArtifact , DataIngestionArtifact
from src.constants import SCHEMA_FILE_PATH


class DataValidation:

    def __init__(self , data_ingestion_artifact : DataIngestionArtifact , data_validation_config : DataValidationConfig):

        try :
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

        except Exception as e:
            raise MyException(e ,sys)
        
    def validate_number_of_columns(self , dataframe : DataFrame) -> bool:
        
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required columns present [{status}]")
            return status
        
        except Exception as e:
            raise MyException(e ,sys)
        
    def is_column_exist(self , df : DataFrame) -> bool :
        try:
            df_columns = df.columns
            missing_numeric_col = []
            missing_categorical_col = []

            for num_col in self._schema_config["numerical_columns"]:
                if num_col not in df_columns:
                    missing_numeric_col.append(num_col)

            if(len(missing_numeric_col) > 0):
                logging.info(f"Missing numerical columns : {missing_numeric_col}")

            for cat_col in self._schema_config["categorical_columns"]:
                if cat_col not in df_columns:
                    missing_categorical_col.append(cat_col)

            if(len(missing_categorical_col) > 0):
                logging.info(f"Missing categorical columns : {missing_categorical_col}")

            return False if len(missing_categorical_col) > 0 or len(missing_numeric_col) > 0 else True
        
        except Exception as e:
            raise MyException(e ,sys)
        
        
    @staticmethod
    def read_data(file_path : str) -> DataFrame:
        try:

            return pd.read_csv(file_path)
        
        except Exception as e:
            raise MyException(e ,sys)


    def initiate_data_validation(self) -> DataValidationArtifact:

        try:

            validation_error_msg  = "" 
            logging.info("Starting data validation. ")
            train_df , test_df = (DataValidation.read_data(file_path= self.data_ingestion_artifact.trained_file_path) ,
                                  DataValidation.read_data(file_path= self.data_ingestion_artifact.test_file_path) )
            
            status = self.validate_number_of_columns(train_df)
            if not status:
                validation_error_msg+= F"Columns are missing in training dataset . "
            else:
                logging.info(f"All columns are present in training dataset .")

            status = self.validate_number_of_columns(test_df)
            if not status:
                validation_error_msg+= F"Columns are missing in test dataset . "
            else:
                logging.info(f"All columns are present in test dataset .")

            status = self.is_column_exist(df=train_df)
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe. "
            else:
                logging.info(f"All categorical/int columns present in training dataframe: {status}")

            status = self.is_column_exist(df=test_df)
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."
            else:
                logging.info(f"All categorical/int columns present in testing dataframe: {status}")

            
            validation_status = len(validation_error_msg) == 0

            data_validation_artifact = DataValidationArtifact(
                validation_status= validation_status,
                message= validation_error_msg,
                validation_report_file_path=  self.data_validation_config.validation_report_file_path
            )

            deport_dir = os.path.dirname(self.data_validation_config.validation_report_file_path)
            os.makedirs(deport_dir , exist_ok= True)

            # JSON file 
            validation_report = {
                "validation_status" : validation_status,
                "validation_message" : validation_error_msg.strip()
            }

            with open(self.data_validation_config.validation_report_file_path, "w") as report_file:
                json.dump(validation_report, report_file, indent=4)

            logging.info("Data validation artifact created and saved to JSON file.")
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e


            

                                  

