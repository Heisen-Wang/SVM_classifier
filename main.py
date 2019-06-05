from Features_extractor import features, train_testfrom Data_reading import data_readfrom sklearn.externals import joblibfrom test import localizationimport seaborn as snsfrom plot import pair_plot, plot_histimport numpy as npimport matplotlib.pyplot as pltfrom keras.models import load_model# read the data, sensor position and target positionstrain, sensor_train, target_train = data_read('./data/piropa_manhattan_900_tx_pos.mat')test, sensor_test, target_test = data_read('./data/piropa_manhattan_25_tx_pos.mat')# creat featuresfeatures_train = features(train)features_test = features(test)# load model and scalermodel = load_model("nn.h5")scaler = joblib.load('scaler.save')# make prediction on test dataX_train, y_train, X_test, y_test = train_test(features_train, features_test)y_pred = model.predict(X_test)# error calculationerror = np.ravel(scaler.inverse_transform(y_pred))-np.ravel(scaler.inverse_transform(y_test))b = scaler.inverse_transform(y_pred)# locationx, rmes_sum, rmes = localization(test, sensor_test, target_test, b)# error plotplot_hist(error)plt.show()