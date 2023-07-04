from functools import reduce

import tensorflow as tf

class ModelFactory():
    def __init__(self,input_dim):
        self.models = []
        self.batches = []
        self.optimizer = []

        self.input_dim = input_dim
        self.createModels()

    def createModels(self): 

        ## Model 1
        self.batches.append(50)
        self.optimizer.append(tf.keras.optimizers.legacy.Adam(learning_rate=0.001))

        self.models.append(tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(self.input_dim[0], self.input_dim[1])),
            tf.keras.layers.Dense(16, activation="relu"),
            #tf.keras.layers.Dense(16),
            tf.keras.layers.Dense(10, activation="softmax")
        ]))

        ## Model 2
        self.batches.append(100)
        self.optimizer.append(tf.keras.optimizers.legacy.Adam(learning_rate=0.0003))
        
        self.models.append(tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.input_dim[0], self.input_dim[1], 1)),
        
            tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3)),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            #tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3)),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            #tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3)),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            #tf.keras.layers.Dropout(0.15),
        
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
        
            tf.keras.layers.Dense(10, activation="softmax")
        ]))


        # Model 3
        self.batches.append(100)
        self.optimizer.append(tf.keras.optimizers.legacy.Adam(learning_rate=0.0003))

        model3 = tf.keras.models.Sequential()
        model3.add(tf.keras.layers.InputLayer(input_shape=(self.input_dim[0], self.input_dim[1], 1)))

        if(self.input_dim[0]==1):
            model3.add(tf.keras.layers.UpSampling2D(size=(2,2)))

        model3.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3),padding='same'))
        model3.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        if(self.input_dim[0]==1):
            model3.add(tf.keras.layers.UpSampling2D(size=(2,2)))

        model3.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),padding="same"))
        model3.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        if(self.input_dim[0]==1):
            model3.add(tf.keras.layers.UpSampling2D(size=(2,2)))

        model3.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),padding="same"))
        model3.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        previousLayer_output_dim = model3.layers[-1].output_shape
        prevLayer_total_output_size = reduce(lambda x, y: x * y if y is not None else x, previousLayer_output_dim, 1)
        nextLayer_input_dim = int(prevLayer_total_output_size/64)

        model3.add(tf.keras.layers.Reshape(target_shape=(nextLayer_input_dim,  64)))
        model3.add(tf.keras.layers.LSTM(32))
        model3.add(tf.keras.layers.Dense(10, activation="softmax"))

        self.models.append(model3)

