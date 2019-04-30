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
test.delay = [item[0
              ] for item in test.delay]

data_train['b'] = train.delay - train.Distance/(3*pow(10,8))
data_test['b'] = test.delay - test.Distance/(3*pow(10,8))

#print(data_test.b)
#print(len(data_train.b))
from sklearn import svm
import numpy as np

#X_train = np.asarray(data_train['b']).reshape(-1, 1)
#X_test = np.asarray(data_test['b']).reshape(-1, 1)
#y_train = np.ravel(data_train.label)
#y_test = np.ravel(data_test.label)
#clf = svm.SVC(kernel='rbf', gamma ='scale')
#clf.fit(X_train, y_train)

X1_train = np.array(data_train['kurtosis']).reshape(-1,1)
X2_train = np.array(data_train['received_energy']).reshape(-1,1)
X1_test = np.array(data_test['kurtosis']).reshape(-1,1)
X2_test = np.array(data_test['received_energy']).reshape(-1,1)
X_train = np.c_[X1_train, X2_train]
X_test = np.c_[X1_test, X2_test]
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
print(y_pred)

b_hat = []
for i in range(len(data_test['b'])):
    if y_pred[i] == 'LOS':
        b_hat.append(0)
    else:
        b_hat.append(data_test['b'][i])
error = b_hat-data_test['b']
e = []
print(error[1])
print(error[0])
for temp in error:
    e.append(temp[0])
print(e[0])
#error = list(map(int, np.log10(e)))
import matplotlib.pyplot as plt
plt.figure(2)
plt.hist(e, 20)
plt.show()
