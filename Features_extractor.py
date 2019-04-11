import math
import numpy as np
import pandas as pd

def db2power(db):
    if len(db) != 0:
        power = pow(10,-db/10)
    else:
        power = np.array([])
    return power
def power2db(power):
    db = 10*math.log10(power)
    return db
def received_energy(power):
    if len(power) != 0:
        e_s = 2.004*sum(power)
    else:
        e_s = np.array([])
    return e_s
def maximum_amplitude(power):
    if len(power) != 0:
        max_p = max(power)
    else:
        max_p = np.array([])
    return max_p
def mean_delay(power, delay):
    if len(power) != 0:
        toi = sum(power*delay)/sum(power)
    else:
        toi = np.array([])
    return toi
def delay_spread(power, delay):
    if len(power) != 0:
        toi_m = mean_delay(power, delay)
        toi_rms = np.sqrt(sum((delay-toi_m)**2*power)/sum(power))
    else:
        toi_rms = np.array([])
    return toi_rms
def kurtosis(power):
    if len(power) != 0:
        mu = 1/len(power)*sum(np.sqrt(power))
        sigma = 1/len(power)*(sum(np.sqrt(power)-mu)**2)
        k = 1/(sigma**2*len(power))*(sum(np.sqrt(power)-mu)**4)
        return k
    else:
        return np.array([])

def EuclideanDistances(A, B):
    BT = B.transpose()
    vecProd = A * BT
    SqA =  A.getA()**2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    SqB = B.getA()**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    ED = (SqED.getA())**0.5
    return np.matrix(ED)

def features(data):
    features_ = {}
    features_head = ['received_energy', 'Max_amplitude', 'mean_excess_delay', 'delay_spread', 'kurtosis',]

    data_temp = data
    power = db2power(data_temp.power)
    delay = data_temp.delay
    dic_temp = {'received_energy': [], 'Max_amplitude': [], 'mean_excess_delay': [], 'delay_spread': [],
                'kurtosis': []}
    for j in range(len(power)):
        dic_temp[features_head[0]].append(power2db(received_energy(power[j])))
        dic_temp[features_head[1]].append(maximum_amplitude(power[j])[0])
        dic_temp[features_head[2]].append(mean_delay(power[j], delay[j])[0])
        dic_temp[features_head[3]].append(delay_spread(power[j], delay[j])[0])
        dic_temp[features_head[4]].append(kurtosis(power[j])[0])
    df = pd.DataFrame(dic_temp)
    features_= df

    features_['label'] = data.los
    # features
    # sort features according to the value of labels
    features_.sort_values('label', inplace=True)
    # replace label with nlos or los
    features_['label'] = features_['label'].replace(0, 'LOS')
    features_['label'] = features_['label'].replace(1, 'NLOS')

    # features['Distance_dif'] = features.mean_excess_delay*3e8-data.Distance
    # fill the NAN value
    features_['kurtosis'] = features_['kurtosis'].where(features_['kurtosis'] > 0)
    features_['kurtosis'] = features_['kurtosis'].fillna(0)
    # sort the value according to the label
    features_.sort_values('label', inplace=True)
    return (features_)