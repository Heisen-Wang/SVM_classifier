from Features_extractor import features
from Data_reading import data_read
from plot import pair_plot

train = data_read('./data/piropa_manhattan_900_tx_pos.mat')
test = data_read('./data/piropa_manhattan_25_tx_pos.mat')

# Generate the train and test data set
data_train = features(train)
data_test = features(test)

pair_plot(data_test)
train.delay = [item[0] for item in train.delay]
test.delay = [item[0] for item in test.delay]

data_train['b'] = train.delay - train.Distance/(3*pow(10,8))
data_test['b'] = test.delay - test.Distance/(3*pow(10,8))

print(data_test)
from sklearn import svm
import numpy as np
X_train = np.asarray(data_train['b']).reshape(-1,1)
X_test = np.asarray(data_test['b']).reshape(-1,1)
y_train = np.ravel(data_train.label)
y_test = np.ravel(data_test.label)
clf = svm.SVC(kernel='rbf', gamma ='scale')
clf.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix,zero_one_loss
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
error_rate= zero_one_loss(y_test, y_pred)
print(error_rate)

