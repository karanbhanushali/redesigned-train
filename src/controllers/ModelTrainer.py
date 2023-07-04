# train different models
import tensorflow as tf

class ModelTrainer():
    def __init__(self):
        self.histories = []

    def trainAllModels(self, models, data, labels):
        for i, model in enumerate(models):
            self.trainModel(data, labels,
                               model,
                               epochs=100,
                               batches=50,
                               optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
                               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                               learning_rate=0.001)


    def trainModel(self, data, label, model, epochs, batches, optimizer, loss, learning_rate):
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=["accuracy"]
        )

        model.summary()
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)
        # fit - trains models
        history = model.fit(data, label, validation_split=0.1, verbose=1, epochs=epochs, batch_size=batches, callbacks=[callback])
        self.histories.append(history)



    # def fit(self):
    #     myInstane = MyClassOfHistory()
    #     myInstane.doStuff()

    #     return myInstane

