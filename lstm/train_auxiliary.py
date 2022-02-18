from operator import ge, index
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


def getData(df, columns, days_before=30, return_all=True):
    '''
    读取原始数据，并生成训练样本
    df             : 原始数据
    column         : 要处理的列
    train_end      : 训练集的终点
    days_before    : 多少天来预测下一天
    return_all     : 是否返回所有数据，默认 True
    generate_index : 是否生成 index
    '''
    cpi_series = df[columns[0]].copy()
    sent_series = df[columns[1]].copy()
    stock_series = df[columns[2]].copy()

    # Build Training Data
    day_val = []

    # for i in range(len(cpi_series)-days_before):
    #     for j in range(days_before, len(cpi_series)):
    #         a = []
    #         a.append(cpi_series[j-days_before: j])
    #         a = np.vstack((a, sent_series[j-days_before: j]))
    #         a = np.vstack((a, stock_series[j-days_before: j]))
    #     day_val.append(a)
    #     print(len(day_val))
    
    # train_input = np.array(day_val)
    train_input = np.load(file="train_input.npy")
    train_input = train_input.reshape(-1, 30, 3)
    
    # Build Training Label
    day_label = []

    # for i in range(len(cpi_series)-days_before):
    #     for j in range(days_before, len(cpi_series)):
    #         a = []
    #         a.append(cpi_series[j])
    #         a = np.vstack((a, sent_series[j]))
    #         a = np.vstack((a, stock_series[j]))
    #     day_label.append(a)
    #     print(len(day_label))
    
    # train_label = np.array(day_label)
    train_label = np.load(file="train_label.npy")
    train_label = train_label.reshape(-1, 1, 3)
    print(train_input.shape, train_label.shape)

    train_data = np.concatenate((train_input, train_label), axis=1)
            
    if return_all:
        return train_input, train_label, train_data, df.index.tolist()
    
    return train_data

# Define Training Dataset
class TrainSet(Dataset):
    def __init__(self, data):
        self.data, self.label = data[:, :-1, :].float(), data[:, -1, :].float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

# Build LSTM Model
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=3,   # 输入尺寸为 1，表示一天的数据
            hidden_size=512,
            num_layers=6, 
            batch_first=True)
        
        self.out = nn.Sequential(
            nn.Linear(512,1))
        
    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)   # None 表示 hidden state 会用全 0 的 state
        out = self.out(r_out[:, -1, :])          # 取最后一天作为输出
        
        return out

# Define Hyper-parameters
LR = 0.0001
EPOCH = 100
DAYS_BEFORE = 30

# Load Dataset
def Load_data():
    df = pd.read_csv('data_all.csv', index_col=0)
    columns = ['CPI', 'Sentiment', 'Adj Close']
    train_input, train_label, train_data, df_index = getData(df, columns=columns, days_before=DAYS_BEFORE)
    return train_input, train_data, train_label, df_index

# df_test = pd.read_csv('apple_raw_test.csv', index_col=0)
# len_test = len(df_test.index) - len(df.index)

# test_data = df_test.filter(['Adj Close'])[-len_test:]
# test_data_label = df_test['Adj Close'].copy()[-len_test:]

# Normalization
def normalization(train_data):
    train_data_numpy = np.array(train_data)
    train_mean = np.mean(train_data_numpy)
    train_std  = np.std(train_data_numpy)
    train_data_numpy = (train_data_numpy - train_mean) / train_std
    train_data_tensor = torch.Tensor(train_data_numpy)
    return train_data_tensor, train_mean, train_std

# Create dataloader
train_input, train_data, train_label, df_index = Load_data()
print('----------Finished Data Loading------------')
train_data_tensor, train_mean, train_std = normalization(train_data)
train_set = TrainSet(train_data_tensor)
train_loader = DataLoader(train_set, batch_size=10, shuffle=True)

# Building Model
rnn = LSTM()
# rnn = torch.load('rnn.pkl')

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.MSELoss()

def train():
    for step in range(EPOCH):
        for tx, ty in train_loader:    
            print(tx.shape)     
            output = rnn(tx)
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
    # test_data = df_test.filter(['Adj Close'])
    prediction, error = test()
    test_data.loc[:, 'Predictions'] = prediction

    # Visualize the data
    totalSeed = test_data.index.tolist()
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