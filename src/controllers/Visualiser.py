import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score,f1_score

class Visualiser():
    def __init__(self):
        pass

    def visualise(self, data, index, labels):
        firstExample = data[index]
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        featuresToVisualize = range(firstExample.shape[1])[6:9]
        featureNames = [
            "Loudness_sma3", "alphaRatio_sma3", "hammarbergIndex_sma3", "slope0-500_sma3", "slope500-1500_sma3",
            "spectralFlux_sma3", "mfcc1_sma3", "mfcc2_sma3", "mfcc3_sma3", "mfcc4_sma3", "mfcc5_sma3", "mfcc6_sma3",
            "mfcc7_sma3", "mfcc8_sma3", "mfcc9_sma3", "mfcc10_sma3", "mfcc11_sma3", "mfcc12_sma3", "mfcc13_sma3",
            "mfcc14_sma3", "mfcc15_sma3", "mfcc16_sma3", "mfcc17_sma3", "mfcc18_sma3", "mfcc19_sma3", "mfcc20_sma3",
            "F0semitoneFrom27.5Hz_sma3nz", "jitterLocal_sma3nz", "shimmerLocaldB_sma3nz", "HNRdBACF_sma3nz",
            "logRelF0-H1-H2_sma3nz", "logRelF0-H1-A3_sma3nz", "F1frequency_sma3nz", "F1bandwidth_sma3nz",
            "F1amplitudeLogRelF0_sma3nz", "F2frequency_sma3nz", "F2bandwidth_sma3nz", "F2amplitudeLogRelF0_sma3nz",
            "F3frequency_sma3nz", "F3bandwidth_sma3nz", "F3amplitudeLogRelF0_sma3nz"
        ]

        for featureIndex in featuresToVisualize:
            ax.plot(range(firstExample.shape[0]), firstExample[:, featureIndex])
            ax.legend(featureNames[6:9])

        ax.set_xlabel("Timeframe")
        ax.set_ylabel("Feature value")
        ax.set_title(f"Example: {index} - Label: {labels[index]}")

        plt.show()

    def _plotHistory(self, history):
        fig = plt.figure(figsize=(10, 5))
        for idx, key in enumerate(['loss', 'accuracy']):
            ax = fig.add_subplot(1, 2, idx + 1)
            plt.plot(history.history[key])
            plt.plot(history.history['val_{}'.format(key)])
            plt.title('model {}'.format(key))
            plt.ylabel(key)
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')

        plt.show()


    def _plotHistories(self, list_of_histories):
        num_models = len(list_of_histories)
        fig, axes = plt.subplots(num_models, 2, figsize=(10, 3 * num_models))

        for i, history in enumerate(list_of_histories):
            ax_loss = axes[i, 0]
            ax_accuracy = axes[i, 1]

            ax_loss.plot(history.history['loss'], label='Training Loss')
            ax_loss.plot(history.history['val_loss'], label='Validation Loss')
            ax_loss.set_xlabel('Epochs')
            ax_loss.set_ylabel('Loss')
            ax_loss.set_title('Model {} - Loss'.format(i + 1))
            ax_loss.legend()

            ax_accuracy.plot(history.history['accuracy'], label='Training Accuracy')
            ax_accuracy.plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax_accuracy.set_xlabel('Epochs')
            ax_accuracy.set_ylabel('Accuracy')
            ax_accuracy.set_title('Model {} - Accuracy'.format(i + 1))
            ax_accuracy.legend()

        plt.tight_layout()
        plt.show()

    def visualizeModelHistory(self, histories):
        if len(histories) > 1:
            self._plotHistories(histories)

        else:
            self._plotHistory(histories[0])

    def visualizeConfusionMatrix(self, predictions, labels):
        predictions = [np.argmax(item) for item in predictions]
        print(predictions)
        print(f"F1: {accuracy_score(predictions, labels)}")
        print(f"F1: {confusion_matrix(predictions, labels)}")
        ConfusionMatrixDisplay.from_predictions(labels, predictions)