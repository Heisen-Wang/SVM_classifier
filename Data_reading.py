import scipy.io as scio
import numpy as np
import pandas as pd
from Features_extractor import EuclideanDistances

def data_read(dataloc):
    data_file = scio.loadmat(dataloc)
    df_head =  ['n_cluster','AoA','EoA','AoD','EoD','delay','power','los','bs_id']

    target_pos = np.matrix(data_file['par'][0][0][8][0][0][0])
    Sensor_pos = [item[0][0] for item in data_file['par'][0][0][11][0]]
    Sensor_pos = np.matrix(Sensor_pos)

    D = EuclideanDistances(Sensor_pos, target_pos)
    Distance = D.getA()

    data_lis = []
    Distance_temp = []
    for i in range(len(data_file['bulk_parameters'])):
        data_tempt = data_file['bulk_parameters'][i]
        dic_temp = {}
        for key in df_head:
            dic_temp[key] = [item[0][0][df_head.index(key)] for item in data_tempt[0][0]
                            if len(item[0]) != 0]
        for item in data_tempt[0][0]:
            if len(item[0]) is not 0:
                Distance_temp.append(Distance[i][item[0][0][-1][0][0]-1])
        df_temp = pd.DataFrame(dic_temp)
        data_lis.append(df_temp)

    # concanating dataframe
    df_data = pd.concat(data_lis, ignore_index=True)
    df_data['Distance'] = Distance_temp
    # drop the empty value
    data = df_data.where(df_data.n_cluster != 0).dropna()
    # reset the index
    data = data.reset_index(drop=True)
    # calculate the label
    data.los = data.los.apply(np.sum)
    data.delay = data.delay.map(lambda delay: delay[0])
    data.bs_id = data.bs_id.astype(int)

    return data, Sensor_pos, target_pos

def to_regular(test, sensor_test, target_test, features_test):
    target_len = test.bs_id.unique()
    sensor_len = sensor_test.shape[0]
    columns_ind = [item+1 for item in range(sensor_len)]

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

