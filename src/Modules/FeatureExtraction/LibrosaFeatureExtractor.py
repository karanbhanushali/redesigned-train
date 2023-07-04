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

class LibrosaFeatureExtractor:
    def __init__(self) -> None:

        # librosa
        self.config = ConfigManager()
        self.helpers = Helpers()
        self.n_mfcc = self.config.librosaConfig.n_mfcc
        self.n_bands = self.config.librosaConfig.n_bands
        self.includeZCR = self.config.librosaConfig.includeZCR
        self.includeSpectralRollOff = self.config.librosaConfig.includeSpectralRollOff
        self.includeSpectralCentroid = self.config.librosaConfig.includeSpectralCentroid
        self.includeMFCCs = self.config.librosaConfig.includeMFCCs
        self.includeContrast = self.config.librosaConfig.includeContrast
        self.includeFlatness = self.config.librosaConfig.includeFlatness
        self.includeRMS = self.config.librosaConfig.includeRMS
        self.outputJsonFile_librosa = f"{self.config.librosaConfig.outputFolder}/{self.config.librosaConfig.outputJsonFile}"
        self.MFCCFeatures = [f"MFCCs_{i}" for i in range (1,self.n_mfcc+1)] 
        self.BandsFeatures = [f"Contrast_{i}" for i in range (1,self.n_bands+2)]
 

    def extract_features(self,files):

        if(self.helpers.files_exists([self.config.librosaConfig.outputFolder+self.config.librosaConfig.outputJsonFile])):
            print("\n Librosa feature Files Exists No Extraction Necessary!")
            return

        print("\nExtracting Librosa Features")    
        grouped_data = {}
        for audio_path in tqdm(files):
            y, sr = librosa.load(audio_path)
            name = audio_path.split('/')[-1]
            grouped_data[name] = self.getFeatures(y,sr) 

        with open(self.outputJsonFile_librosa, 'w') as file:
            json.dump(grouped_data, file)
        return grouped_data
    
    def getFeatures(self,y,sr):
        dictionaryForAFile ={}
        if(self.includeZCR):
            zeroCrossingRate = librosa.feature.zero_crossing_rate(y)  
            dictionaryForAFile["ZeroCrossingRate"] = zeroCrossingRate[0].tolist()
        if(self.includeSpectralRollOff):
            spectral_rolloff = librosa.feature.spectral_rolloff(y, sr=sr) 
            dictionaryForAFile["SpectralRolloff"] = spectral_rolloff[0].tolist()
        if(self.includeSpectralCentroid):
            spectral_centroid = librosa.feature.spectral_centroid(y, sr=sr) 
            dictionaryForAFile["SpectralCentroid"] = spectral_centroid[0].tolist()
        if(self.includeMFCCs):
            mfccs = librosa.feature.mfcc(y, sr=sr,n_mfcc=self.n_mfcc)           
            for index,value in enumerate(self.MFCCFeatures):
                dictionaryForAFile[value] = mfccs[index].tolist()
        S = np.abs(librosa.stft(y)) 
        if(self.includeContrast):
            contrast = librosa.feature.spectral_contrast(S=S, sr=sr,n_bands=self.n_bands)  
            for index,value in enumerate(self.BandsFeatures):
                dictionaryForAFile[value] = contrast[index].tolist()
        if(self.includeFlatness):
            flatness = librosa.feature.spectral_flatness(S=S)  
            dictionaryForAFile["Flatness"] = flatness[0].tolist()
        if(self.includeRMS):
            rms = librosa.feature.rms(y=y)
            dictionaryForAFile["RMS"] = rms[0].tolist()
        return dictionaryForAFile

## Librosa End