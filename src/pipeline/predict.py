import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline():
    def __init__(self) -> None:
        pass 
    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData():
    def __init__(self,
                 carat:float,
                 x:float,
                 y:float,
                 z:float,
                 depth:float,
                 table:float,
                 cut:str,
                 color:str,
                 clarity:str,
                 ) -> None:
        self.carat=carat

        self.x=x
        self.y=y
        self.z=z
        self.depth=depth
        self.table=table
        self.cut=cut
        self.color=color
        self.clarity=clarity


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "carat": [self.carat],
                "x": [self.x],
                "y": [self.y],
                "z": [self.z],
                "depth": [self.depth],
                "table": [self.table],
                "cut": [self.cut],
                "color": [self.color],
                "clarity": [self.clarity],
            }
            return pd.DataFrame(custom_data_input_dict)
    
        except Exception as e:
            raise CustomException(e,sys) 
        
