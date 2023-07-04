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

class PretrainedFeatureExtractor:
    def __init__(self) -> None:

        # librosa
        self.config = ConfigManager()
        self.helpers = Helpers()
 

    def extract_features(self,files):

        if(self.helpers.files_exists([self.config.wav2vecConfig.outputFolder + '/'+self.config.wav2vecConfig.outputJsonFile])):
            print(" \nPreTrained feature Files Exists No Extraction Necessary!")
            return


        print("\nExtracting Wav2Vec Features")    
        csv_file_path = self.config.wav2vecConfig.outputFolder + self.config.wav2vecConfig.outputJsonFile#,"Data/Normalized/facebook_wav2vec/normalizedData.csv"
        label_file = self.config.wav2vecConfig.outputFolder + "/labels.csv"
        desired_sample_rate = 16000
        max_duration = 1.0  # sec   ## need to check data
        model_checkpoint = "facebook/wav2vec2-base"
        batch_size = 32
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
#        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        list_of_features = []
        list_of_labels = []
        data_prep = PrepareData()
        for file in tqdm(files):
        #    print(file)
            try:
                label = data_prep.extract_data_from_name(file)
            except:
                continue
            try:
                waveform, sample_rate = torchaudio.load(file)
                if sample_rate != 16000:
                    resampler = Resample(orig_freq=sample_rate, new_freq=16000)
                    waveform = resampler(waveform)
                    sample_rate = 16000
#                waveform, sample_rate = torchaudio.load(file, normalize=True, channels_first=True, sample_rate=desired_sample_rate)
            #    print(waveform)
                features = feature_extractor(waveform, sampling_rate=sample_rate, return_tensors="pt")
            except Exception as ex:
            #    print(ex)
                continue
            #print(features)
            list_of_labels.append(label)
#            print((features['input_values']))
#            print(type(features['input_values']))
            list_of_features.append([tensor.squeeze().numpy() for tensor in features['input_values']])
        # print(list_of_features)
        # print((list_of_features[0]))
        # print(type(list_of_features[0]))
        # print(type(list_of_features[0][0]))
        # print(len(list_of_features[0][0]))
        # max_length = max(len(arr[0]) for arr in list_of_features)
        # padded_list = [np.pad(arr, (0, max_length - len(arr)), mode='constant') for arr in list_of_features]
        # np.savetxt(csv_file_path, np.vstack(padded_list), delimiter=',')
        with open(label_file, "w") as file:
            for item in list_of_labels:
                file.write(str(item) + "\n")
        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)
        with open(csv_file_path, 'a') as f:
            for array in tqdm(list_of_features):
                np.savetxt(f, [array[0]], delimiter=',', fmt='%.7f')
 
 
 
