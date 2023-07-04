import os 
import random
import json
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import itertools
from src.utils.ConfigManager import *
import glob
class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
    

class Helpers():
    def __init__(self) -> None:
        pass
    
    def getFiles(self,source)->list:
        folder_path = source
        file_extension = '**/*.wav'
        file_paths = glob.glob(f"{folder_path}/{file_extension}", recursive=True)
        return file_paths  
    
    def files_exists(self,file_names):
 
        for file_path in file_names:
            if os.path.exists(file_path):
                continue
            else:
                return False
        return True