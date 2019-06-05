from Features_extractor import features, db2power
from Data_reading import data_read
import numpy as np

def localization(data, sensor_pos, target_pos, bias):

    pd = features(data)
    idx = data.bs_id.unique() - 1
    data['energy'] = pd['received_energy']
    data['b'] = bias
    dis = dict.fromkeys(idx, [])
    sigma = dict.fromkeys(idx, [])
    sigma_total = dict.fromkeys(idx, [])
    x = dict.fromkeys(idx, [])
    b = dict.fromkeys(idx, [])
    rmes = dict.fromkeys(idx, [])
    for id in idx:
        #dis[id] = data['Distance'].loc[data['bs_id'] == id + 1]
        dis[id] = data['delay'].loc[data['bs_id'] == id+1]*(3*pow(10,8))
        dis[id] = dis[id].reset_index(drop=True)
        sigma[id] = -data['energy'].loc[data['bs_id'] == id + 1] + (-139)
        sigma[id] = sigma[id].reset_index(drop=True)
        sigma[id] = db2power(-sigma[id])
        sigma_total[id] = 1 / sum(1/sigma[id])
        #b[id] = data['b'].loc[data['bs_id'] == id + 1]
        b[id] = data['b'].loc[data['bs_id'] == id + 1]
        b[id] = b[id].reset_index(drop=True)

    for id in idx:
        dis_temp = dis[id] - b[id]
        sigma_temp = sigma[id]
        iter = 0
        x[id] = np.array([0, 0, 0])
        while iter < 20000:
            x_new = 0
            for i in range(len(sensor_pos)):
                norm = np.linalg.norm(sensor_pos[i] - x[id])
                sensor = x[id] - sensor_pos[i]
                if sensor_pos is not x[id]:
                    x_new = x_new + sigma_total[id]/sigma_temp[i] * (dis_temp[i] * (sensor / norm) + sensor_pos[i])
                else:
                    x_new = x_new + sigma_total[id]/sigma_temp[i] * (dis_temp[i] * np.transpose(np.array([1, 0])))
            x[id] = x_new
            iter += 1
        print(x[id])
        rmes[id] = np.linalg.norm(x[id] - target_pos[id])
    return x, sum(rmes.values())/len(idx), rmes


data, sensor_pos, target_pos = data_read('./data/piropa_manhattan_25_tx_pos.mat')



