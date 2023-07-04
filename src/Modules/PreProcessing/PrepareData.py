import os
from src.utils.ConfigManager import *
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split
class PrepareData():
    def __init__(self) -> None:
        self.extractors = ['openSmile_Functional','openSmile_LLD','librosa','facebook_wav2vec']
        self.config = ConfigManager()
        self.label_list = []
    def padding(self,data):
        maxLen = 0
        newData = []

        for example in data:
            if len(example) > maxLen:
                maxLen = len(example)
 

        for example in data:
            if len(example) == maxLen:
                newData.append(example)
                continue

            paddingSize = (maxLen - len(example))
            padding = paddingSize * [len(example[0]) * [0]]

            example.extend(padding)
            newData.append(example)

        return newData


    def normalize(self,data):
        for i,feature in enumerate(data[0][0]): # i=41 features
            feature = []
            for example in data: #[[[1,2],[3,4],[5,6],[7,8]],   [[],[],],   [[],[],[],[],],   [[],[],[],[],[],[]]]
                for tf in example:
                    feature.append(tf[i])

            minValue = min(feature)
            maxValue = max(feature)
            valueRange = maxValue - minValue

            for j, example in enumerate(data): # j = 1940
                for k, tf in enumerate(example): # k = 62
                    newValue = (tf[i] - minValue)/valueRange
                    data[j][k][i] = newValue

        return data

    def normalize_and_pad_data(self):
        for extractor in self.extractors:
            self.prepareData(extractor)
    def prepareData(self,extractor:str=''):
        normalizedFile = f'Data/Normalized/{extractor}/normalizedData.csv'
        if os.path.isfile(normalizedFile):
            return
        
        print(f'\nNormalizing Data for -- {extractor}')
        if not os.path.exists(f'Data/Normalized/{extractor}'):
            os.makedirs(f'Data/Normalized/{extractor}',exist_ok=True)

        if(extractor == "facebook_wav2vec"):
            wav2vec_FeatureFile = self.config.wav2vecConfig.outputFolder+self.config.wav2vecConfig.outputJsonFile
            self.pad_csv_data(wav2vec_FeatureFile,normalizedFile)
            return
        data, labels = self.getdata(extractor)
        normalizedData = self.normalize(data)
        paddedData = self.padding(normalizedData)

        data = np.array(paddedData, dtype=float)
        labels = np.array(labels, dtype=int)
 

        a = len(data)
        b = len(data[0])
        c = len(data[0][0])
 


        df = pd.DataFrame(data.reshape(a, b*c))
       
        df.to_csv(normalizedFile, header=False, index=False )

        dfLabels = pd.DataFrame(labels)
        dfLabels.to_csv(f'Data/Normalized/{extractor}/labels.csv', header=False, index=False)

        dfShape = pd.DataFrame([a,b,c])
        dfShape.to_csv(f"Data/Normalized/{extractor}/dataShape.csv", header=False, index=False)

 

    def getdata(self,extractor:str):
    
        librosa_FeatureFile = self.config.librosaConfig.outputFolder+self.config.librosaConfig.outputJsonFile
        openSmileLLD_FeatureFile = self.config.openSmileConfig.LLDOutputFolder+'/'+self.config.openSmileConfig.outputJsonFile
        openSmileFunctional_FeatureFile = self.config.openSmileConfig.FunctionalsOuputFolder + '/'+self.config.openSmileConfig.outputJsonFile
        wav2vec_FeatureFile = self.config.wav2vecConfig.outputFolder+self.config.wav2vecConfig.outputJsonFile  
        self.label_list = []

        if(extractor == "openSmile_Functional"):
            file = openSmileFunctional_FeatureFile
        elif(extractor == "openSmile_LLD"):
            file = openSmileLLD_FeatureFile
        elif(extractor == "librosa"):
            file = librosa_FeatureFile
        elif(extractor == "facebook_wav2vec"):
            return
        else:
            raise('Invalid Extractor')
            return

        df = pd.read_json(file).T      
        shuffled_df = df.sample(frac=1, random_state=42)
        
 
        features_list = []
        for index,file in  tqdm(shuffled_df.iterrows(),total=shuffled_df.shape[0]):
            data = file

            name = index 
            try:
                self.label_list.append(self.extract_data_from_name(name))
            except:
                continue
            if('Name' in data): 
                data = data.drop(['Name'])
            if('Timestamp' in data):
                data = data.drop(['Timestamp'])
            sd = data.apply(pd.Series)
            li_ti = sd.T.values.tolist()
            float_list = [[float(x) for x in sublist] for sublist in li_ti]
            features_list.append(float_list)


        return features_list,self.label_list


    def extract_data_from_name(self,name:str):
        name_without_extention = name.split('.')[0]
        split_string = [name_without_extention[i:i+2] for i in range(0, len(name_without_extention), 2)] # maybe used later when encoding region name as feature
        return int(name_without_extention[-1])

    def pad_csv_data(self,inputFile,outputFile):
        with open(inputFile, 'r') as input_file:
            reader = csv.reader(input_file)
            rows = list(reader)
        output_list = [[float(element) for element in sublist] for sublist in rows]    
        max_length = max(len(sublist) for sublist in output_list)
        padded_list = [sublist + [0.0] * (max_length - len(sublist)) for sublist in tqdm(output_list)]
        with open(outputFile, 'w', newline='') as output_file:
            writer = csv.writer(output_file)
            writer.writerows(padded_list)
        dfShape = pd.DataFrame([len(padded_list),1,max_length])
        shutil.copy2("Data/wav2vec/labels.csv","Data/Normalized/facebook_wav2vec/")
        dfShape.to_csv(f"Data/Normalized/facebook_wav2vec/dataShape.csv", header=False, index=False)

    def split_dataset_into_splits(self,extractor='',path_to_features:str = '',path_to_labels:str= ''):
        folderPath = f'Data/Normalized/{extractor}'
        path_to_features = folderPath+'/normalizedData.csv'
        path_to_labels = folderPath+'/labels.csv'
        path_to_shapes = folderPath+'/dataShape.csv'
        labelArray = []
    
        featuresDF = pd.read_csv(path_to_features,sep=',',header=None,dtype=float)
    
        shapes = pd.read_csv(path_to_shapes, header=None).values.flatten()
        with open(path_to_labels, 'r') as file:
            lines = file.readlines()
        labelArray = [int(line.strip()) for line in lines]
        label_df = pd.DataFrame(labelArray)
        X = featuresDF
        y = label_df
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        training_data = {"data":X_train.values.reshape(len(X_train),shapes[1],shapes[2]),"labels":y_train.values.flatten()}
        testing_data = {"data":X_test.values.reshape(len(X_test),shapes[1],shapes[2]),"labels":y_test.values.flatten()}
        return training_data,testing_data 
