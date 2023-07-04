from src.controllers.ModelTrainer import *
import json
import time

class ModelLogger():
    def __init__(self):
        self.logDict = {"models": []}


    def prepareModelInfo(self, models, histories):

        modelList = []
        for model, history in zip(models, histories):
            jmodel = model.to_json()
            data = json.loads(jmodel)
            config = data.get("config")

            historyDict = history.history
            loss = historyDict["loss"][-1]
            accuracy = historyDict["accuracy"][-1]
            val_loss = historyDict["val_loss"][-1]
            val_accuracy = historyDict["val_accuracy"][-1]

            modelInfo = {
                "timestamp": time.time(),
                "loss": loss,
                "val_loss": val_loss,
                "accuracy": accuracy,
                "val_accuracy": val_accuracy,
                "class_name": data.get("class_name"),
                "config": config,

            }

            modelList.append(modelInfo)


        self.logDict["models"] = modelList


    def saveIntoFile(self):
        with open('logs/log.txt', 'a') as file:
            for model in self.logDict["models"]:
                file.write(str(model)+"\n")
