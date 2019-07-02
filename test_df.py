from Features_extractor import features,train_test
from Data_reading import data_read, to_regular
import numpy as np
from plot import plot_reg
import pandas as pd
test, sensor_test, target_test = data_read('./data/piropa_manhattan_25_tx_pos.mat')
train, sensor_train, target_train = data_read('./data/piropa_manhattan_900_tx_pos.mat')
features_test = features(test)
features_train = features(train)
features_ori = features(test)
features_test = features_test.loc[features_test['label'] == 'LOS'].drop(['label'], axis = 1).astype(float)

features_train = features_train.loc[features_train['label'] == 'LOS'].drop(['label'], axis = 1).astype(float)

import keras
from keras.models import Model, Sequential
from keras.layers import Dense,  Dropout
from Features_extractor import features, train_test
from IPython.display import clear_output
from test import localization
from sklearn.externals import joblib
import seaborn as sns


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

# sns.pairplot(features_test, size = 1.5)
# plt.savefig('./fig/pairplot_reg.png')

from sklearn.preprocessing import StandardScaler, MinMaxScaler
X_test = features_test[['received_energy', 'Max_amplitude', 'mean_excess_delay', 'delay_spread', 'kurtosis']].as_matrix()
X_train = features_train[['received_energy', 'Max_amplitude', 'mean_excess_delay', 'delay_spread', 'kurtosis']].as_matrix()
y_test = features_test['b'].as_matrix()
y_train = features_train['b'].as_matrix()

scaler = MinMaxScaler()
X_test = scaler.fit_transform(X_test)
X_train = scaler.fit_transform(X_train)

model = Sequential()
model.add(Dense(10, input_shape=(X_train.shape[1],)))
model.add(Dense(256, kernel_initializer= 'normal',activation='selu'))
model.add(Dense(1, kernel_initializer= 'normal', activation='elu'))
model.summary()

# compile
model.compile(optimizer='Adagrad', loss='mse')

# train
model.fit(X_train, y_train, batch_size=200,  validation_split=0.2,
          epochs=400, shuffle=True)
# error calculation
y_pred = model.predict(X_test)
# scaler = joblib.load('scaler.save')
# #error = np.ravel(scaler.inverse_transform(y_pred))-np.ravel(scaler.inverse_transform(y_test))
error = y_pred-y_test.reshape(-1,1)

new_df = pd.DataFrame({'b_est': y_pred.tolist()}, index = features_ori.index[features_ori['label'] == 'LOS'])
test['b_est'] = 0
test.update(new_df)
#localization(test, sensor_test, target_test, test['b_est'])
plot_reg(test)
