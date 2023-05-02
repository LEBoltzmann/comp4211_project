import numpy as np
import pandas as pd
import torch as tc
import torch.nn as nn
from torch.utils.data import DataLoader, dataset
import dataview
import modelMake


#constants config
input_size = 10
hidden_size = 1
batch_size = 1


data = pd.read_csv('./flowdb/0101007040B_flow.csv')
data = data.dropna()
using_para = ['p01m', 'tmpf', 'dwpf', 'height', 'cfs']


dataview.dataStand(data, using_para)
print(data.head(5))

split_data = dataview.dataSplit(data)
train_ds = split_data[ : 500]
test_ds = split_data[500 : ]



model = modelMake.RNN(input_size, hidden_size, batch_size)
criterion = nn.MSELoss()
optimizer = tc.optim.Adam(model.parameters(), lr=0.01)


for epoch in range(100):
    loss = 0.0
    print('Epoch [{}/{}] training'.format(epoch+1, 100))
    for dataframe in train_ds:
        data_train = modelMake.MyDataset(dataframe[using_para])
        dataloader = DataLoader(data_train, batch_size = batch_size, shuffle = False)

        for data in dataloader:
            model.train()
            x = data[0].to(tc.float32)
            y = data[1].to(tc.float32)
    
            out = model(x)
            out = out.squeeze(0)

            loss_i = criterion(out, y)
    
            optimizer.zero_grad()
            loss_i.backward()
            loss += loss_i.item()
            optimizer.step()
    
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 100, loss))
    tc.save(model.state_dict(), 'model_2.pth')
