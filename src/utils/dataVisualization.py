import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def visualise(data, index, labels):
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


def plotHistory(history):
    fig = plt.figure(figsize=(10, 5))
    for idx, key in enumerate(['loss', 'accuracy']):
        ax = fig.add_subplot(1, 2, idx+1)
        plt.plot(history.history[key])
        plt.plot(history.history['val_{}'.format(key)])
        plt.title('model {}'.format(key))
        plt.ylabel(key)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')

    plt.show()