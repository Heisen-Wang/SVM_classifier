import keras
from keras.models import Model, Sequential
from keras.layers import Dense,  Dropout
from Features_extractor import features, train_test
from Data_reading import data_read, to_regular
from IPython.display import clear_output
from test import localization
from sklearn.externals import joblib
import seaborn as sns
from keras.optimizers import RMSprop
from plot import plot_hist

import numpy as np
import matplotlib.pyplot as plt


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        #plt.legend()
        plt.show()
plot_losses = PlotLosses()

train, sensor_train, target_train = data_read('./data/piropa_manhattan_900_tx_pos.mat')
test, sensor_test, target_test = data_read('./data/piropa_manhattan_25_tx_pos.mat')

# Generate the train and test data set
data_train = to_regular(train, sensor_train, target_train,features(train))
data_test = to_regular(test, sensor_test, target_test, features(test))
data_train.to_pickle('./data/data_train')
data_test.to_pickle('./data/data_test')
X_train, y_train, X_test, y_test = train_test(data_train, data_test)

# build the model

model = Sequential()
model.add(Dropout(0.2, input_shape=(X_train.shape[1],)))
model.add(Dense(32, kernel_initializer= 'normal',activation='relu'))
model.add(Dense(64, kernel_initializer= 'normal',activation='relu'))
model.add(Dense(128, kernel_initializer= 'normal',activation='relu'))
model.add(Dense(256, kernel_initializer= 'normal',activation='relu'))
model.add(Dense(128, kernel_initializer= 'normal',activation='relu'))
model.add(Dense(64, kernel_initializer= 'normal',activation='relu'))
model.add(Dense(1, kernel_initializer= 'normal', activation='tanh'))
model.summary()

# compile
model.compile(optimizer='adam', loss='mse')

# train
model.fit(X_train, y_train, batch_size=200, validation_data= (X_test, y_test),
          epochs=500, shuffle=True, callbacks=[plot_losses])
model.save('nn.h5')
# test
# print('\nTesting ------------')
# loss, accuracy = model.evaluate(X_test, y_test)
#
# print('test loss: ', loss)
# print('test accuracy: ', accuracy)
