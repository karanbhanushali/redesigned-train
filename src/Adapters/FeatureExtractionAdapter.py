import os
from src.utils.ConfigManager import *
from src.Modules.FeatureExtraction.LibrosaFeatureExtractor import *
from src.Modules.FeatureExtraction.OpenSmileFeatureExtractor import *
from src.Modules.FeatureExtraction.PretrainedFeatureExtractor import *
from src.utils.dataVisualization import *
from src.utils.Helpers import *
class FeatureExtractionAdapter():
    def __init__(self) -> None:
        self.config = ConfigManager()
        self.dataPreparation = PrepareData()
        self.helpers = Helpers()
        self.openSmileFeatureExtractor = OpenSmileFeatureExtractor()
        self.librosaFeatureExtractor = LibrosaFeatureExtractor()
        self.PretrainedFeatureExtractor = PretrainedFeatureExtractor()
        self.extractor = self.config.dataConfig.featureExtractor
    def get_data(self):

        if(self.config.appConfig.performFeatureExtraction):            
            self.extract_features()
        else:
            print("\nNo Feature Extraction would take place -- Flag is OFF!!\n")
        self.dataPreparation.normalize_and_pad_data()
        return self.dataPreparation.split_dataset_into_splits(self.extractor)
    
        
    def extract_features(self):
        files = self.helpers.getFiles(self.config.dataConfig.dataSource)
        self.openSmileFeatureExtractor.extract_features(files)
        self.librosaFeatureExtractor.extract_features(files)
        self.PretrainedFeatureExtractor.extract_features(files)

    