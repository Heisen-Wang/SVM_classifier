import scipy.io as scio
import numpy as np
import pandas as pd
from Features_extractor import EuclideanDistances

def data_read(dataloc):
    data_file = scio.loadmat(dataloc)
    df_head = ['n_cluster', 'AoA', 'EoA', 'AoD', 'EoD', 'delay', 'power', 'los', 'bs_id']

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

    #concanating dataframe
    df_data = pd.concat(data_lis, ignore_index=True)
    df_data['Distance'] = Distance_temp
    #drop the empty value
    data = df_data.where(df_data.n_cluster != 0).dropna()
    #reset the index
    data = data.reset_index(drop=True)
    #calculate the label
    data.los = data.los.apply(np.sum)
    return data

