import numpy as np
import pandas as pd
import torch as tc
import torch.nn as nn
from torchsummary import summary

#define rnn model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size = self.input_size, hidden_size = self.hidden_size)
        self.linear_1 = nn.Linear(3, 30)
        self.linear_3 = nn.Linear(30, 1)

    def forward(self, x, y):
        # RNN layer
        out, _= self.rnn(x)
        out = self.linear_1(tc.cat([out, y], dim = 1))
        out = self.linear_3(out)

        #linear layer
        
        return out

#define data set
class MyDataset(tc.utils.data.Dataset):
    def __init__(self, data, train_len):
        self.data = data
        self.train_len = train_len
        
    def __len__(self):
        return self.data.shape[0] - self.train_len
    
    def __getitem__(self, index):
        before = self.data[index : index + self.train_len, 2]
        current = self.data[index + self.train_len - 1, 0:2]
        target = self.data[index + self.train_len, 2]
        out = (before, current, target)
        return out



#main area
if __name__ == '__main__':
    print("start main")

    data_1 = pd.read_csv('./flowdb/01010500FVE_flow.csv', low_memory=False)

    print(data_1.shape)
    data_1 = data_1.drop_duplicates().dropna()
    print(data_1.shape)
    print(data_1.loc[:, ['p01m', 'tmpf', 'cfs']].corr())

    #data standardize
    p01m = data_1['p01m'].to_numpy()
    mean_p01m = np.mean( p01m )
    std_p01m = np.std( p01m )
    data_1['p01m'] = (p01m - mean_p01m) / std_p01m

    tmpf = data_1['tmpf'].to_numpy()
    mean_tmpf = np.mean( tmpf )
    std_tmpf = np.std( tmpf )
    data_1['tmpf'] = (tmpf - mean_tmpf) / std_tmpf

    cfs = data_1['cfs'].to_numpy()
    mean_cfs = np.mean( cfs )
    std_cfs = np.std( cfs )
    data_1['cfs'] = (cfs - mean_cfs) / std_cfs

    #define constants
    train_len = 5
    shape = data_1.shape
    train_shape = int( np.ceil( shape[0] * 0.8 ) )
    hidden_size = 1

    #spilt the data used out
    train_ds = tc.tensor( data_1.loc[ :, ['p01m', 'tmpf', 'cfs']].iloc[ : train_shape, :].to_numpy()
        )
    test_ds = tc.tensor( data_1.loc[ : , ['p01m', 'tmpf', 'cfs']].iloc[ train_shape:, :].to_numpy()
        )
    
    #initialization
    model = RNN(train_len, hidden_size)
    criterion = nn.MSELoss()
    optimizer = tc.optim.Adam(model.parameters(), lr=0.01)

    data_train = MyDataset(train_ds, train_len)
    data_test = MyDataset(test_ds, train_len)
    dataloader = tc.utils.data.DataLoader(data_train, batch_size=1, shuffle=False)

    #train section
    for epoch in range(100):
        loss = 0.0
        print('Epoch [{}/{}] training'.format(epoch+1, 100))
        for i, data in enumerate(dataloader):
            model.train()
    
D           out = model(data[0].to(tc.float32), data[1].to(tc.float32))
    
            loss_i = criterion(out, data[2].to(tc.float32))
    
            optimizer.zero_grad()
            loss_i.backward()
            loss += loss_i.item()
            optimizer.step()
    
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 100, loss))
        tc.save(model.state_dict(), 'model_2.pth')

    #save model

    #evaluation
    model.eval()
    loss = 0.0
    with tc.no_grad():
        for i, data in test_ds:
            output = model(data[0].to(tc.float32), data[1].to(tc.float32))
            loss += ( output.item()-data[2].item() ) ** 2

    print('loss on test set: %.2f%%' % loss)

