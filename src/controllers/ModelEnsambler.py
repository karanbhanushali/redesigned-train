import numpy as np

class ModelEnsambler():
    def __init__(self, models):
        self.models = models

    def predict(self, data):
        predictions = [model.predict(data) for model in self.models]

        # which model has the highest confidence value
        maxValue = None
        model = None
        for i, prediction in enumerate(predictions):
            maxPrediction = np.max(prediction)
            if maxValue is None or maxPrediction > maxValue:
                maxValue = maxPrediction
                model = i


        return predictions[model] # prediction at index model is the best