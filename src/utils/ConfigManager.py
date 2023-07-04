import os.path

from src.utils.Helpers import *
import json

class ConfigManager(metaclass=SingletonMeta):
    def __init__(self):
        config = self.load()

        self.appConfig = AppConfiguration(config["appConfig"])
        self.dataConfig = DataConfiguration(config["dataSource"])
        self.openSmileConfig = OpenSmileConfiguration(config["openSmileConfig"])
        self.librosaConfig = LibrosaConfiguration(config["librosaConfig"])
        self.wav2vecConfig = Wav2Vec(config["wav2vec"])
    def load(self):
        with open("Configurations/appConfig.json") as file:
            return json.load(file)


class AppConfiguration():
    def __init__(self, configSection) -> None:
        self.logFolder = configSection['logFolder']
        self.logFile = configSection['logFile']
        self.logLevel = configSection['logLevel']
        self.performFeatureExtraction = configSection['performFeatureExtraction']


class DataConfiguration():
    def __init__(self, configSection) -> None:
        self.dataSource = configSection['folderPath']
        self.featureExtractor = configSection['featureExtractor']
        self.trainDataPath = configSection["trainDataPath"]


class OpenSmileConfiguration():
    def __init__(self, configSection) -> None:
        self.smileExtractPath = configSection['smileExtractPath']
        self.smileConfigPath = configSection['smileConfigPath']
        self.FunctionalsOuputFolder = configSection['FunctionalsOuputFolder']
        self.LLDOutputFolder = configSection['LLDOutputFolder']
        self.outputCSVFile = configSection['outputCSVFile']
        self.outputJsonFile = configSection['outputJsonFile']
        self.featuresToExtract = configSection['featuresToExtract']
        self.LLDFeatures = configSection['LLDFeatures']
        self.FunctionalFeatures = configSection['FunctionalFeatures']


class LibrosaConfiguration():
    def __init__(self, configSection) -> None:
        self.n_mfcc = int(configSection['n_mfcc'])
        self.n_bands = int(configSection['n_bands'])
        self.includeZCR = configSection['includeZCR']
        self.includeSpectralRollOff = configSection['includeSpectralRollOff']
        self.includeSpectralCentroid = configSection['includeSpectralCentroid']
        self.includeMFCCs = configSection['includeMFCCs']
        self.includeContrast = configSection['includeContrast']
        self.includeFlatness = configSection['includeFlatness']
        self.includeRMS = configSection['includeRMS']
        self.outputJsonFile = configSection['outputJsonFile']
        self.outputFolder = configSection['outputFolder']

class Wav2Vec():
    def __init__(self, configSection) -> None:
        self.outputJsonFile = configSection['outputJsonFile']
        self.outputFolder = configSection['outputFolder']
