import os
import sys

import dill
import yaml 

import numpy as np
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging

def read_yaml_file(file_path : str) -> dict:
    try :
        with open(file_path , "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
        
    except Exception as e:
        raise MyException(e , sys)
    
def write_yaml_file(file_path : str , content : object , replace : bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path) , exist_ok= True)

        with open(file_path , "w") as yaml_file:
            yaml.dump(content , yaml_file)
            
    except Exception as e:
        raise MyException(e , sys) from e
    
def load_object(file_path : str) -> object:

    try :
        with open(file_path , "rb") as  file_obj:
            obj = dill.load(file_obj)
        return obj

    except Exception as e:
        raise MyException(e , sys) from e

def save_numpy_array_data(file_path : str , arr : np.array) :

    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name , exist_ok= True)

        with open(file_path, "wb") as file_obj:
            np.save(file_obj , arr)
    
    except Exception as e:
        raise MyException(e , sys) from e
    
def load_numpy_array_data(file_path : str) -> np.array :

    try:
        with open(file_path , "rb") as file_obj:
            return np.load(file_obj)
    
    except Exception as e:
        raise MyException(e , sys) from e
    
def save_object(file_path : str , obj : object) -> None :
    logging.info("Entered save_object method of utils ")

    try:
        os.makedirs(os.path.dirname(file_path) , exist_ok= True)
        with open(file_path , "wb") as file_obj:
            dill.dump(obj , file_obj)

        logging.info("Exited the save_object method .")

    except Exception as e:
        raise MyException(e , sys) from e

    

    


