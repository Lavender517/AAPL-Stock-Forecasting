from operator import ge, index
from runpy import run_path
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from pandas.plotting import autocorrelation_plot


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
    #     a = []
    #     a.append(cpi_series[i: i+days_before])
    #     a = np.vstack((a, sent_series[i: i+days_before]))
    #     a = np.vstack((a, stock_series[i: i+days_before]))
    #     day_val.append(a)
    #     print(len(day_val))
    
    # train_input = np.array(day_val)
    # train_input = np.swapaxes(train_input,1,2)
    # np.save(file="train_input.npy", arr=train_input)
    train_input = np.load(file="./data/train_input.npy")
    
    # Build Training Label
    day_label = []

    # for i in range(len(cpi_series)-days_before):
    #     a = []
    #     a.append(cpi_series[i+days_before])
    #     a = np.vstack((a, sent_series[i+days_before]))
    #     a = np.vstack((a, stock_series[i+days_before]))
    #     day_label.append(a)
    #     print(len(day_label))
    
    # train_label = np.array(day_label)
    # train_label = np.swapaxes(train_label,1,2)
    # np.save(file="train_label.npy", arr=train_label)
    train_label = np.load(file="./data/train_label.npy")

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
class LSTM_aux(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_variables, if_train=True):
        super(LSTM_aux, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.if_train = if_train
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_variables)
        
    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)   # None 表示 hidden state 会用全 0 的 state
        # print(r_out.shape) # [10, 30, 64]
        # if self.if_train:
        #     output = self.fc(r_out) # [10, 30, 3]
        #     print(output.shape)
        # else:
        output = self.fc(r_out[:, -1, :]) # [10, 3]
        
        return output

# Define Hyper-parameters
LR = 0.0001
EPOCH = 100
DAYS_BEFORE = 30
columns = ['CPI', 'Sentiment', 'Adj Close']

# Load Dataset
def Load_data():
    df = pd.read_csv('./data/data_all.csv', index_col=0)
    train_series = df[columns].values
    train_input, train_label, train_data, df_index = getData(df, columns=columns, days_before=DAYS_BEFORE)
    return df, train_input, train_data, train_label, train_series, df_index

df_test = pd.read_csv('./data/data_all_test.csv', index_col=0)
test_data = df_test.filter(['Adj Close'])
test_label = df_test['Adj Close'].copy()

# Normalization
def normalization(train_data):
    train_data_numpy = np.array(train_data)
    train_mean = np.mean(train_data_numpy)
    train_std  = np.std(train_data_numpy)
    train_data_numpy = (train_data_numpy - train_mean) / train_std
    train_data_tensor = torch.Tensor(train_data_numpy)
    return train_data_tensor, train_mean, train_std

# Create dataloader
df, train_input, train_data, train_label, train_series, df_index = Load_data()
print('----------Finished Data Loading------------')
train_data_tensor, train_mean, train_std = normalization(train_data)
train_set = TrainSet(train_data_tensor)
train_loader = DataLoader(train_set, batch_size=10, shuffle=True)

# Building Model
# rnn = LSTM_aux(input_size=3, hidden_size=64, num_layers=1, num_variables=3, if_train=True)
rnn = torch.load('./rnn_model/rnn_aux_1.75.pkl')

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.MSELoss() # Mean Squared Error
loss_func_MAE = nn.L1Loss(reduction='mean') # Mean Absolute Error

def train():
    for step in range(EPOCH):
        for tx, ty in train_loader:  
            output = rnn(tx)
            loss = loss_func(output, ty)
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # back propagation, compute gradients
            optimizer.step()
        print(step, loss.cpu())
        if step % 10:
            torch.save(rnn, './rnn_model/rnn_aux.pkl')
    torch.save(rnn, './rnn_model/rnn_aux.pkl')

def test():
    predictions = []

    # Do the Normalization to all data
    scaled_train_series = (train_series - train_mean) / train_std
    scaled_train_tensor = torch.Tensor(scaled_train_series)

    for i in range(len(scaled_train_series), len(scaled_train_series)+len(test_data)):
        x = scaled_train_tensor[i - DAYS_BEFORE:i]
        # 将 x 填充到 (bs, ts, is) 中的 timesteps
        x = torch.unsqueeze(x, dim=0) # x.shape = [1, 30, 1]
        # print(x.shape)
        y = rnn(x)
        # print(y.shape) #[1, 3]

        # print(scaled_train_tensor.shape) # [n(1027-1046), 3]
        scaled_train_tensor = torch.cat((scaled_train_tensor, y), 0)
        predict_stock = torch.squeeze(y.cpu())[-1]
        predictions.append(predict_stock.detach().numpy() * train_std + train_mean)

    # error = loss_func(torch.Tensor(predictions), torch.Tensor(test_label.values))
    # error = loss_func_MAE(torch.Tensor(predictions), torch.Tensor(test_label.values))
    error = r2_score(predictions, test_label.values)
    print(error.item())
    return predictions, error

def result_plot():
    train_plot = df.filter(['Adj Close'])
    predictions, error = test()
    test_data.loc[:, 'Predictions'] = predictions

    # Visualize the data
    all_data = pd.read_csv('./data/apple_raw_all_test.csv', index_col=0)
    totalSeed = all_data.index.tolist()
    xticks=list(range(0,len(totalSeed), 88))
    xlabels=[totalSeed[x] for x in xticks]
    xticks.append(len(totalSeed))
    xlabels.append(totalSeed[-1])

    plt.figure(figsize=(16,6))
    plt.title('Multiparameter Model Prediction Results')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price for AAPL Stock', fontsize=18)
    plt.plot(train_plot['Adj Close'])
    plt.plot(test_data[['Adj Close', 'Predictions']])
    ax = plt.gca()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    plt.legend(['Train', 'Real-Test', 'Predictions'], loc='lower right')
    plt.grid()
    plt.savefig('./pictures/result_auxiliary.jpg')

def residual_distribution():
    predictions, _ = test()
    # calculate residuals
    residuals = [test_label[i]-predictions[i] for i in range(len(predictions))]
    residuals = pd.DataFrame(residuals)
    print(residuals.describe())
    # residuals.plot()
    residuals.hist() # histogram plot
    # plt.show()
    residuals.plot(kind='kde') # density plot
    autocorrelation_plot(residuals)
    plt.show()

if __name__ == "__main__":
    # train()
    # result_plot()
    residual_distribution()