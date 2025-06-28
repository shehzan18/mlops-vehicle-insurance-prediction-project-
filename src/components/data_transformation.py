import os 
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler , StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTEENN

from src.constants import TARGET_COLUMN , SCHEMA_FILE_PATH , CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact , DataValidationArtifact , DataTransformationArtifact
from src.logger import logging
from src.exception import MyException
from src.utils.main_utils import read_yaml_file , save_numpy_array_data ,save_object

class DataTransformation:

    def __init__(self,
                 data_ingestion_artifact : DataIngestionArtifact,
                 data_validation_artifact : DataValidationArtifact ,
                 data_transformation_config : DataTransformationConfig):
        
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(file_path= SCHEMA_FILE_PATH)

        except Exception as e:
            raise MyException(e , sys)
        
    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e , sys)
        
    def get_data_transformer_object(self) -> Pipeline:

        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            numeric_transformer = StandardScaler()
            min_max_scaler = MinMaxScaler()
            logging.info("Transformers Initialized: StandardScaler-MinMaxScaler")

            num_features = self._schema_config["num_features"]
            mm_columns = self._schema_config["mm_columns"]
            logging.info("Cols loaded from schema.")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("StandardScaler" , numeric_transformer , num_features),
                    ("MinMaxScaler" , min_max_scaler , mm_columns)
                ],

                    remainder='passthrough'  # leave the  other columns as they are
                
            )

            final_pipeline = Pipeline(steps= [("Preprocessor" , preprocessor)])
            logging.info("Final Pipeline Ready!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise MyException(e, sys) from e
        
    def _map_gender_column(self , df : DataFrame) -> DataFrame:
        logging.info("Mapping 'Gender' column to binary values")
        df['Gender'] = df["Gender"].map({"Female" : 0 , "Male" : 1}).astype(int)
        return df
    
    def _rename_columns(self , df : DataFrame) -> DataFrame:
        logging.info("Renaming specific columns and casting to int")

        df = df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })

        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype(int)

        return df
    
    def _create_dummy_columns(self, df : DataFrame) -> DataFrame:
        """Create dummy variables for categorical features."""
        logging.info("Creating dummy variables for categorical features")
        df = pd.get_dummies(df, drop_first=True)
        return df
    
    def drop_columns(self , df: DataFrame) -> DataFrame:
        logging.info("Dropping 'id' column")
        drop_col = self._schema_config["drop_columns"]
        
        if drop_col in df.columns:
            df = df.drop(drop_col, axis=1)
        return df
    

    def initiate_data_transformation(self ) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation Started !!!")

            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)
            
            train_df = self.read_data(file_path= self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path= self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            input_feature_train_df = train_df.drop(columns = [TARGET_COLUMN] , axis =1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns = [TARGET_COLUMN] , axis =1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Input and Target cols defined for both train and test df.")

            input_feature_train_df = self._map_gender_column(input_feature_train_df)
            input_feature_train_df = self.drop_columns(input_feature_train_df)
            input_feature_train_df = self._create_dummy_columns(input_feature_train_df)
            input_feature_train_df = self._rename_columns(input_feature_train_df)

            input_feature_test_df = self._map_gender_column(input_feature_test_df)
            input_feature_test_df = self.drop_columns(input_feature_test_df)
            input_feature_test_df = self._create_dummy_columns(input_feature_test_df)
            input_feature_test_df = self._rename_columns(input_feature_test_df)
            logging.info("Custom transformations applied to train and test data")

            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            logging.info("Initializing transformation for Training-data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Initializing transformation for Testing-data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Transformation done end to end to train-test df.")

            logging.info("Applying SMOTEENN for handling imbalanced dataset.")
            smt = SMOTEENN(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr, target_feature_test_df
            )
            logging.info("SMOTEENN applied to train-test df.")

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            logging.info("feature-target concatenation done for train-test df.")

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, arr =train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,arr = test_arr)
            logging.info("Saving transformation object and transformed files.")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
        
        except Exception as e:
            raise MyException(e , sys)
        
        




                       


        

        


        

