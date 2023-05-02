import pandas as pd
import numpy as np

def dataSplit(data):
    cut_point = 0
    split_data = []

    for i in range( len(data) - 1 ):
        this = data['Unnamed: 0'].iloc[i]
        nex = data['Unnamed: 0'].iloc[i + 1]
        delta = nex - this

        #cut if follow requirement
        if delta > 2:
            pre = cut_point
            cut_point = i

            if cut_point - pre >= 100:
                print("===split section from {} to {}===".format(pre, cut_point))
                split_data.append(data.iloc[pre : cut_point + 1, : ])

    return split_data

def dataStand(data, para):
    for column in para:
        print('===standlizing data {} with type {}'.format(column, type(data[column].to_numpy()[0])))
        this = data[column].to_numpy()
        mean = np.mean(this)
        std = np.std(this)
        data[column] = ( this - mean ) / std

if __name__ == '__main__':

    data = pd.read_csv('./flowdb/0101007040B_flow.csv')

    data = data.dropna()
    using_para = ['p01m', 'tmpf', 'dwpf', 'height', 'cfs']
    dataStand(data, using_para)
    print(data.head(5))

    split_data = dataSplit(data)
    print(len(split_data))

    for data in split_data:
        print('{} '.format(len(data)))


