from Features_extractor import features
from Data_reading import data_read
import numpy as np
import pandas as pd
test, sensor_test, target_test = data_read('./data/piropa_manhattan_25_tx_pos.mat')
features_test = features(test)

def to_regular(test, sensor_test, target_test, features_test):
    target_len = test.bs_id.unique()
    sensor_len = sensor_test.shape[0]
    columns_ind = ['s'+str(item+1) for item in range(sensor_len)]

    arrays = [target_len, np.array(['pos', 'distance','received_energy', 'Max_amplitude', 'mean_excess_delay',
                        'delay_spread', 'kurtosis', 'delay','bias','b_est','label'])
              ]
    index = pd.MultiIndex.from_product(arrays, names=['first', 'second'])

    df = pd.DataFrame(np.zeros((index.shape[0], sensor_len)), columns=columns_ind, index=index)
    df = df.astype(object)

    ind = 0
    for i in columns_ind:
        for t_i in target_len:
            df[i][t_i]['pos'] = target_test[t_i]
            df[i][t_i]['distance'] = test['Distance'][ind]
            df[i][t_i]['delay'] = test['delay'][ind]
            df[i][t_i]['received_energy'] = features_test['received_energy'][ind]
            df[i][t_i]['Max_amplitude'] = features_test['Max_amplitude'][ind]
            df[i][t_i]['mean_excess_delay'] = features_test['mean_excess_delay'][ind]
            df[i][t_i]['delay_spread'] = features_test['delay_spread'][ind]
            df[i][t_i]['kurtosis'] = features_test['kurtosis'][ind]
            df[i][t_i]['label'] = features_test['label'][ind]
            df[i][t_i]['bias'] = features_test['b'][ind]
            ind += 1
    return df

df = to_regular(test, sensor_test, target_test, features_test)
