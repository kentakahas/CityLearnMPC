import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt


class DNN:
    def __init__(self, n_feats, n_targets):
        self.n_feats = n_feats
        self.n_targets = n_targets

    def generator(self, hp):
        model = Sequential()
        model.add(InputLayer(input_shape=(self.n_feats,)))

        # Setup DNN layers
        dnn_layers = hp.Int('DNN layers', min_value=2, max_value=6)
        add_batch_norm = hp.Boolean("Batch normalization")

        # DNN model to analyze the input parameters
        for i in range(dnn_layers):
            model.add(Dense(units=hp.Int('units ' + str(i + 1),
                                         min_value=16,
                                         max_value=256,
                                         step=16),
                            kernel_initializer='he_normal',
                            kernel_regularizer=regularizers.l2(0.001)))
            model.add(Activation('selu'))
            model.add(Dropout(rate=hp.Float('dropout ' + str(i + 1),
                                            min_value=0.1,
                                            max_value=0.8,
                                            step=0.1)))
            if add_batch_norm:
                model.add(BatchNormalization())

        model.add(Dense(self.n_targets, activation='linear'))

        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate=0.001,
                                         beta_1=0.9,
                                         beta_2=0.999,
                                         amsgrad=True)

        model.compile(optimizer=self.optimizer,
                      loss='mse')

        return model

    def tune_model(self, X, y, trials=50):
        # Run bayesian optimization
        self.tuner = kt.BayesianOptimization(self.generator,
                                             objective='val_loss',
                                             max_trials=trials,
                                             overwrite=True)

        es = EarlyStopping(monitor='val_loss',
                           min_delta=0.01,
                           patience=50,
                           mode='min')

        self.tuner.search(X,
                          y,
                          epochs=500,
                          batch_size=32,
                          validation_split=0.25,
                          callbacks=[es])

        best_hp = self.tuner.get_best_hyperparameters()[0]
        self.model = self.tuner.hypermodel.build(best_hp)

    def fit(self, X, y, epoch=3000):
        self.optimizer.learning_rate.assign(0.0005)

        es = EarlyStopping(monitor='val_loss',
                           min_delta=0.001,
                           patience=epoch / 10,
                           mode='min',
                           verbose=1)

        self.checkpoint_filepath = 'DNN.ckpt'

        mc = ModelCheckpoint(filepath=self.checkpoint_filepath,
                             save_weights_only=True,
                             monitor='val_loss',
                             save_best_only=True)

        self.hist = self.model.fit(X, y,
                                   epochs=epoch,
                                   batch_size=1024,
                                   verbose=2,
                                   validation_split=0.25,
                                   callbacks=[es, mc])

    def visualize(self):
        loss = self.hist.history['loss']
        val_loss = self.hist.history['val_loss']

        plt.plot(range(len(loss)), loss, label='loss')
        plt.plot(range(len(val_loss)), val_loss, label='val_loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

    def predict(self, X):
        self.model.load_weights(self.checkpoint_filepath)
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)

        rmse = []
        eep = []

        for i in range(len(y_pred)):
            rmse.append(np.sqrt(mean_squared_error(y_test[i], y_pred[i])))
            eep.append(rmse[i] / y_test.max() * 100)

        return rmse, eep

    def print_eval(self, X_test, y_test):
        rmse, eep = self.evaluate(X_test, y_test)

        print("Min RMSE: {:.2f}, Max RMSE: {:.2f}, Average RMSE: {:.2f}".format(min(rmse), max(rmse), np.mean(rmse)))
        print("Min EEP: {:.2f}%, Max EEP: {:.2f}%, Average EEP: {:.2f}%".format(min(eep), max(eep), np.mean(eep)))

    def plot_eval(self, X_test, y_test):
        rmse, eep = self.evaluate(X_test, y_test)

        fig, axs = plt.subplots(2, figsize=(9, 4))
        axs[0].hist(rmse, bins=100)
        axs[0].set_title('RMSE')
        axs[1].hist(eep, bins=100, density=True)
        axs[1].set_title('EEP')

        plt.tight_layout()
        plt.show()

    def save(self, cwd, filename):
        json = os.path.join(cwd, f'models/{filename}_DNN.json')
        hdf5 = os.path.join(cwd, f'models/{filename}_DNN.h5')

        model_json = self.model.to_json()
        with open(json, 'w') as json_file:
            json_file.write(model_json)
        self.model.save_weights(hdf5)
