import os 
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    


def evaluate_models(xtrain,ytrain,xtest,ytest,models,params):
    try:
        report={}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=params[list(models.keys())[i]]
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(xtrain,ytrain)

            model.set_params(**gs.best_params_)

            model.fit(xtrain, ytrain)
            y_train_pred = model.predict(xtrain)
            y_test_pred= model.predict(xtest)
            r2_train = r2_score(ytrain, y_train_pred)
            r2_test = r2_score(ytest, y_test_pred)
            
            report[list(models.keys())[i]]=r2_test
        
        return report

    except Exception as e:
        raise CustomException(e,sys)



def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)