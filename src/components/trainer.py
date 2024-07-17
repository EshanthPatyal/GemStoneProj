import os 
import sys
from dataclasses import dataclass

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts",'model.pkl')

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,xtrain,ytrain,xtest,ytest):
        try:
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                # "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            params={
                "Linear Regression":{},
                "Lasso":{},
                "Ridge":{},
                "K-Neighbors Regressor":{},
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest Regressor":{
                    'n_estimators': [8,16,32]
                },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                # "CatBoosting Regressor":{
                #         'depth': [4,5,6,7,8,9, 10],
                #         'learning_rate' : [0.01,0.02,0.03,0.04],
                #         'iterations'    : [300,400,500,600]},

                "AdaBoost Regressor":{
                },

            }
            model_report:dict=evaluate_models(xtrain=xtrain,ytrain=ytrain,xtest=xtest,ytest=ytest,models=models,params=params)
            
            best_score=max(sorted(model_report.values()))
            
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_score)
                ]
            best_model=models[best_model_name]

            if(best_score<0.6):
                raise CustomException("No Best Model Found (acc<0.6)")

            logging.info("Found Best model")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(xtest)

            r2_square= r2_score(ytest,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
