from operator import ge
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from data_acquisition import get_stock_data


def getData(df, column, days_before=30, return_all=True):
    '''
    读取原始数据，并生成训练样本
    df             : 原始数据
    column         : 要处理的列
    train_end      : 训练集的终点
    days_before    : 多少天来预测下一天
    return_all     : 是否返回所有数据，默认 True
    generate_index : 是否生成 index
    '''
    train_series = df[column].copy()
    
    # 创建训练集
    train_data = pd.DataFrame()
        
    # 通过移位，创建历史 days_before 天的数据
    for i in range(days_before):
        # 当前数据的 7 天前的数据，应该取 开始到 7 天前的数据； 昨天的数据，应该为开始到昨天的数据，如：
        # [..., 1,2,3,4,5,6,7] 昨天的为 [..., 1,2,3,4,5,6]
        # 比如从 [2:-7+2]，其长度为 len - 7
        train_data['x%d' % i] = train_series.tolist()[i: -days_before + i]
            
    # 获取对应的 label
    train_data['y'] = train_series.tolist()[days_before:]
                
    if return_all:
        return train_data, train_series, df.index.tolist()
    
    return train_data

# Define Training Dataset
class TrainSet(Dataset):
    def __init__(self, data):
        # 定义好 image 的路径
        # data 取前多少天的数据， label 取最后一天的数据
        self.data, self.label = data[:, :-1].float(), data[:, -1].float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

# Build LSTM Model
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=1,   # 输入尺寸为 1，表示一天的数据
            hidden_size=64,
            num_layers=1, 
            batch_first=True)
        
        self.out = nn.Sequential(
            nn.Linear(64,1))
        
    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)   # None 表示 hidden state 会用全 0 的 state
        out = self.out(r_out[:, -1, :])          # 取最后一天作为输出
        
        return out

# Define Hyper-parameters
LR = 0.0001
EPOCH = 100
DAYS_BEFORE = 30

# Implement Trainig Dataset
df = pd.read_csv('apple_raw.csv', index_col=0)
train_data, train_series, df_index = getData(df, 'Adj Close', days_before=DAYS_BEFORE)

df_test = pd.read_csv('apple_raw_test.csv', index_col=0)
len_test = len(df_test.index) - len(df.index)

test_data = df_test.filter(['Adj Close'])[-len_test:]
test_data_label = df_test['Adj Close'].copy()[-len_test:]

# Draw the original stock price
# plt.figure(figsize=(12,8))
# plt.plot(df_index, all_series, label='real-data')

# Normalization
train_data_numpy = np.array(train_data)
train_mean = np.mean(train_data_numpy)
train_std  = np.std(train_data_numpy)
train_data_numpy = (train_data_numpy - train_mean) / train_std
train_data_tensor = torch.Tensor(train_data_numpy)

# Create dataloader
train_set = TrainSet(train_data_tensor)
train_loader = DataLoader(train_set, batch_size=10, shuffle=True)

# Building Model
# rnn = LSTM()
rnn = torch.load('rnn_stock_1.76.pkl')

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.MSELoss()

def train():
    for step in range(EPOCH):
        for tx, ty in train_loader:        
            output = rnn(torch.unsqueeze(tx, dim=2))
            loss = loss_func(torch.squeeze(output), ty)
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # back propagation, compute gradients
            optimizer.step()
        print(step, loss.cpu())
        if step % 10:
            torch.save(rnn, 'rnn.pkl')
    torch.save(rnn, 'rnn.pkl')

def test():
    predict_test = []

    # Do the Normalization to all data
    scaled_train_series = (train_series - train_mean) / train_std
    scaled_train_tensor = torch.Tensor(scaled_train_series)

    for i in range(len(scaled_train_series), len(scaled_train_series)+len_test):
        x = scaled_train_tensor[i - DAYS_BEFORE:i]
        # 将 x 填充到 (bs, ts, is) 中的 timesteps
        x = torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=2) # x.shape = [1, 30, 1]
        y = rnn(x)
        
        print(scaled_train_tensor.shape)
        scaled_train_tensor = torch.cat((scaled_train_tensor, y[0]), 0)
        # scaled_train_tensor.cat()
        predict_test.append(torch.squeeze(y.cpu()).detach().numpy() * train_std + train_mean)

    error = np.sqrt(np.mean(((predict_test - test_data_label) ** 2)))
    print(error)
    return predict_test, error

def result_plot():
    train_plot = df.filter(['Adj Close'])
    prediction, error = test()
    test_data.loc[:, 'Predictions'] = prediction

    # Visualize the data
    totalSeed = df_test.index.tolist()
    xticks=list(range(0,len(totalSeed), 88))
    xlabels=[totalSeed[x] for x in xticks]
    xticks.append(len(totalSeed))
    xlabels.append(totalSeed[-1])

    plt.figure(figsize=(16,6))
    plt.title('Model Prediction Results')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Adjust Close Price USD ($)', fontsize=18)
    plt.plot(train_plot['Adj Close'])
    plt.plot(test_data[['Adj Close', 'Predictions']])
    ax = plt.gca()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    plt.legend(['Train', 'Real-Test', 'Predictions'], loc='lower right')
    plt.grid()
    plt.savefig('./pictures/result.jpg')

if __name__ == "__main__":
    train()
    # result_plot()