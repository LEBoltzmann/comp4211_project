
import numpy as np
import pandas as pd
import torch as tc
import torch.nn as nn
from torch.utils.data import Dataset

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.rnn = nn.RNN(input_size = self.input_size, hidden_size = self.hidden_size)
        self.linear_1 = nn.Linear(4,50)
        self.linear_2 = nn.Linear(50,50)
        self.linear_3 = nn.Linear(50,10)

    def forward(self, x):
        out = self.linear_1(x)
        out = self.linear_2(out)
        out = self.linear_3(out)
        out, _ = self.rnn(out)

        return out


class MyDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        
    def __len__(self):
        return len(self.data )
    
    def __getitem__(self, index):
        x = self.data.iloc[index, :4].to_numpy()
        y = self.data.iloc[index,4]
        return (x, y)




if __name__ == '__main__':
    pass
