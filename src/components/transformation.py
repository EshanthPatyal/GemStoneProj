import os
import sys 
import pandas as pd
import numpy as np 
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from src.utils import save_object


@dataclass
class DataTranformConfig:
    preprocessor_ob_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransform:
    def __init__(self) -> None:
        self.data_tranform_config=DataTranformConfig()

    def get_data_transform_object(self):
        try:
            numeric_features=['carat', 'depth', 'table', 'x', 'y', 'z']
            categorical_features=['cut', 'color', 'clarity']

            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            from sklearn.compose import ColumnTransformer
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OrdinalEncoder,StandardScaler

            # Numerical Pipeline
            num_pipeline = Pipeline(
                            steps = [
                            ('imputer',SimpleImputer(strategy='median')),
                            ('scaler',StandardScaler())
                            ]
                        )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                            steps=[
                            ('imputer',SimpleImputer(strategy='most_frequent')),
                            ('ordinal_encoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                            ('scaler',StandardScaler())
                            ]
                        )

            preprocessor = ColumnTransformer(
                            [
                            ('num_pipeline',num_pipeline,numeric_features),
                            ('cat_pipeline',cat_pipeline,categorical_features)
                            ]
                        )
            return preprocessor
        

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transform(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transform_object()

            target_column_name="price"
            numerical_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                " Preprocessing df "
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_tranform_config.preprocessor_ob_file_path,
                obj=preprocessing_obj

            )

            return (
                input_feature_train_arr,
                target_feature_train_df,
                input_feature_test_arr,
                target_feature_test_df,
                self.data_tranform_config.preprocessor_ob_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        