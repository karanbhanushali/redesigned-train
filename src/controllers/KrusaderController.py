import pandas as pd
import numpy as np
import tensorflow as tf
import os

from src.controllers.ModelEnsambler import ModelEnsambler
from src.controllers.ModelFactory import ModelFactory
from src.utils.ConfigManager import *
from src.controllers.Visualiser import *
from src.controllers.ModelLogger import *
from src.Adapters.FeatureExtractionAdapter import *
class KrusaderController:
    def __init__(self):
        # self.data = None
        # self.labels = None
        self.training_data = {}
        self.testing_data = {}
        self.extractor = 'None'
        self.config = ConfigManager()
        self.featureExtractionAdapter = FeatureExtractionAdapter()

    def start(self):

        self.training_data , self.testing_data = self.featureExtractionAdapter.get_data()

        trainX = self.training_data["data"]
        trainY = self.training_data["labels"]
        testX = self.testing_data["data"]
        testY = self.testing_data["labels"]

        factory = ModelFactory(trainX[0].shape)
        trainer = ModelTrainer()
        trainer.trainAllModels(factory.models, trainX, trainY)

        # Predict

        ensambler = ModelEnsambler(factory.models)
        prediction = ensambler.predict(testX)

        # # fit model no training data
        # model = XGBClassifier()
        # model.fit(trainX, trainY)
        #
        #
        # # make predictions for test data
        # y_pred = model.predict(testY)
        # predictions = [round(value) for value in y_pred]
        #
        #
        # # evaluate predictions
        # accuracy = accuracy_score(testY, predictions)
        # print("Accuracy: %.2f%%" % (accuracy * 100.0))


        # Test Model

        # Perform Evaluation

        # Predict Results

        visualiser = Visualiser()
        visualiser.visualizeConfusionMatrix(prediction, testY)
        visualiser.visualizeModelHistory(trainer.histories)


    def visualise(self):
        pass