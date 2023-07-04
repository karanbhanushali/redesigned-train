import os
import glob
from tqdm import tqdm 
import librosa
import json
from src.utils.ConfigManager import *
import subprocess
import pandas as pd
import numpy as np
import csv
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor
import glob
import torch
import torchaudio
from torchaudio.transforms import Resample
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoFeatureExtractor
from src.Modules.PreProcessing.PrepareData import *
from src.utils.Helpers import *

class OpenSmileFeatureExtractor:
    def __init__(self) -> None:

        self.config = ConfigManager()
        self.helpers = Helpers()
        self.FunctionalFeatures = self.config.openSmileConfig.FunctionalFeatures#.split(',')
        self.FunctionaloutputCSVFile = f"{self.config.openSmileConfig.FunctionalsOuputFolder}/{self.config.openSmileConfig.outputCSVFile}"
        self.FunctionaloutputJsonFile = f"{self.config.openSmileConfig.FunctionalsOuputFolder}/{self.config.openSmileConfig.outputJsonFile}" 
        self.Functionalcommand = "-appendcsv 1 -csvoutput"

        self.LLDFeatures = self.config.openSmileConfig.LLDFeatures#.split(',')
        self.LLDoutputCSVFile = f"{self.config.openSmileConfig.LLDOutputFolder}/{self.config.openSmileConfig.outputCSVFile}"
        self.LLDoutputJsonFile = f"{self.config.openSmileConfig.LLDOutputFolder}/{self.config.openSmileConfig.outputJsonFile}" 
        self.LLDcommand = "-appendcsvlld 1 -lldcsvoutput"


    def extract_features(self,files):

        if(self.helpers.files_exists([self.config.openSmileConfig.LLDOutputFolder+'/'+self.config.openSmileConfig.outputJsonFile,
                      self.config.openSmileConfig.FunctionalsOuputFolder + '/'+self.config.openSmileConfig.outputJsonFile,])):
            print(" \nOpenSmile Functional & LLD feature Files Exists No Extraction Necessary!")
            return

        print("\nExtracting OpenSmile Features -- Functionals & LLDs")    
        smileExtract = self.config.openSmileConfig.smileExtractPath.replace(' ', '\\ ') 
        configLocation = self.config.openSmileConfig.smileConfigPath.replace(' ', '\\ ')            
        LLDoutputFile = self.LLDoutputCSVFile.replace(' ', '\\ ') 
        FunctionaloutputFile = self.FunctionaloutputCSVFile.replace(' ', '\\ ') 

        os.system(f"rm {LLDoutputFile}")
        os.system(f"rm {FunctionaloutputFile}")
        for file in tqdm(files):
            name = file.split('/')[-1].replace(' ', '\\ ') 
            file = file.replace(' ', '\\ ') 
            LLDextractCommand = f"{smileExtract} -C {configLocation} -I {file} {self.LLDcommand} {LLDoutputFile} -instname '{name}'" 
            FunctionalextractCommand = f"{smileExtract} -C {configLocation} -I {file} {self.Functionalcommand} {FunctionaloutputFile} -instname '{name}'" 
            subprocess.run(LLDextractCommand, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(FunctionalextractCommand, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
         
        df = pd.read_csv(self.LLDoutputCSVFile,delimiter=";")
        df.columns = self.LLDFeatures
        df.to_csv(self.LLDoutputCSVFile, index=False)        
        self.convertOpenSmileCSVtoJSON(self.LLDFeatures,self.LLDoutputCSVFile,self.LLDoutputJsonFile)

        df = pd.read_csv(self.FunctionaloutputCSVFile,delimiter=";")
        df.columns = self.FunctionalFeatures 
        df.to_csv(self.FunctionaloutputCSVFile, index=False)        
        self.convertOpenSmileCSVtoJSON(self.FunctionalFeatures,self.FunctionaloutputCSVFile,self.FunctionaloutputJsonFile)

        return True
 
    def convertOpenSmileCSVtoJSON(self,features,file,outputFile):
 
        column_name = features[0]
        with open(file, 'r') as file:
            csv_data = csv.DictReader(file)
            grouped_data = {}
            for row in tqdm(csv_data):
                group_value = row[column_name]
                if group_value not in grouped_data:
                    grouped_data[group_value] = dict.fromkeys(features)
                dictionaryForAFile = grouped_data[group_value]      
                for key, __ in row.items():
                    if(dictionaryForAFile[key]==None):
                        dictionaryForAFile[key] = []
                    dictionaryForAFile[key].append(row[key])
                grouped_data[group_value] = dictionaryForAFile
        with open(outputFile, 'w') as file:
            json.dump(grouped_data, file)
        return grouped_data
 