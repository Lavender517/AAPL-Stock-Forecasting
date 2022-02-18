from operator import ge
from statistics import mean
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter 
from data_acquisition_storage import get_stock_data
from sklearn.metrics import r2_score
from pandas.plotting import autocorrelation_plot


def getData(df, column, days_before=30, return_all=True):
    '''
    Read the original data and generate training samples
    df              : original data
    column          : the column to be processed
    train_end       : End of a training set
    days_before     : How many days to predict the next day
    return_all      : whether to return all data. The default value is True
    generate_index  : indicates whether to generate index
    '''
    train_series = df[column].copy()
    
    # Build Training Dataset
    train_data = pd.DataFrame()
        
    # Create data with a history of days before by shifting
    for i in range(days_before):
        train_data['x%d' % i] = train_series.tolist()[i: -days_before + i]
            
    # Acquire Label
    train_data['y'] = train_series.tolist()[days_before:]
                
    if return_all:
        return train_data, train_series, df.index.tolist()
    
    return train_data

# Define Training Dataset
class TrainSet(Dataset):
    def __init__(self, data):
        self.data, self.label = data[:, :-1].float(), data[:, -1].float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

# Build LSTM Model
class LSTM_model(nn.Module):
    def __init__(self):
        super(LSTM_model, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=1,   # Represents one-day data
            hidden_size=64,
            num_layers=1, 
            batch_first=True)
        
        self.out = nn.Sequential(
            nn.Linear(64,1))
        
    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)  
        out = self.out(r_out[:, -1, :])          # The last day's value is output
        
        return out

# Define Hyper-parameters
LR = 0.0001
EPOCH = 100
DAYS_BEFORE = 30

# Implement Trainig Dataset
df = pd.read_csv('./data/apple_raw.csv', index_col=0)
train_data, train_series, df_index = getData(df, 'Adj Close', days_before=DAYS_BEFORE)

df_test = pd.read_csv('./data/apple_raw_test.csv', index_col=0)
test_data = df_test.filter(['Adj Close'])
test_data_label = df_test['Adj Close'].copy()

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
rnn = LSTM_model()
# rnn = torch.load('./rnn_model/rnn_uni.pkl')
log_writer = SummaryWriter() # Write log file

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.MSELoss() # Mean Squared Error
loss_func_MAE = nn.L1Loss(reduction='mean') # Mean Absolute Error

def train():
    for step in range(EPOCH):
        for tx, ty in train_loader:       
            output = rnn(torch.unsqueeze(tx, dim=2))
            loss = loss_func(torch.squeeze(output), ty)
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # back propagation, compute gradients
            optimizer.step()
        print(step, loss.cpu())
        log_writer.add_scalar('Loss/Train', float(loss), step) # Write in Tensorboard
        if step % 10:
            torch.save(rnn, './rnn_model/rnn.pkl')
    torch.save(rnn, './rnn_model/rnn.pkl')

def valid():
    predictions = []

    # Do the Normalization to all data
    scaled_train_series = (train_series - train_mean) / train_std
    scaled_train_tensor = torch.Tensor(scaled_train_series)

    for i in range(DAYS_BEFORE, len(scaled_train_tensor)):
        x = scaled_train_tensor[i - DAYS_BEFORE:i]
        # Populate x with timesteps in (bs, ts, is)
        x = torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=2)
        y = rnn(x)
        
        predictions.append(torch.squeeze(y.cpu()).detach().numpy() * train_std + train_mean)

    # error = loss_func_MAE(torch.Tensor(predictions), torch.Tensor(train_series[DAYS_BEFORE:]))
    error = r2_score(predictions, train_series[DAYS_BEFORE:])
    print(error.item())
    return predictions, error

def test():
    predict_test = []

    # Do the Normalization to all data
    scaled_train_series = (train_series - train_mean) / train_std
    scaled_train_tensor = torch.Tensor(scaled_train_series)

    for i in range(len(scaled_train_series), len(scaled_train_series)+len(test_data)):
        x = scaled_train_tensor[i - DAYS_BEFORE:i]
        x = torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=2) # x.shape = [1, 30, 1]
        y = rnn(x) # [1, 30, 1]
        
        print(scaled_train_tensor.shape)
        scaled_train_tensor = torch.cat((scaled_train_tensor, y[0]), 0)
        # scaled_train_tensor.cat()
        predict_test.append(torch.squeeze(y.cpu()).detach().numpy() * train_std + train_mean)

    # error = loss_func(torch.Tensor(predict_test), torch.Tensor(test_data_label))
    # error = loss_func_MAE(torch.Tensor(predict_test), torch.Tensor(test_data_label))
    error = r2_score(predict_test, test_data_label)
    print(error.item())
    return predict_test, error

def result_plot():
    train_plot = df.filter(['Adj Close'])
    prediction_test, _ = test()
    test_data.loc[:, 'Predictions'] = prediction_test
    valid_plot = train_plot[DAYS_BEFORE:].copy()
    prediction_valid, _ = valid()
    valid_plot.loc[:, 'Predictions'] = prediction_valid

    # Visualize the data
    all_data = pd.read_csv('./data/apple_raw_all_test.csv', index_col=0)
    totalSeed = all_data.index.tolist()
    xticks=list(range(0,len(totalSeed), 88))
    xlabels=[totalSeed[x] for x in xticks]
    xticks.append(len(totalSeed))
    xlabels.append(totalSeed[-1])

    plt.figure(figsize=(16,6))
    plt.title('Model Validation and Prediction Results')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price for AAPL Stock', fontsize=18)
    plt.plot(train_plot['Adj Close'])
    plt.plot(valid_plot['Predictions'])
    plt.plot(test_data[['Adj Close', 'Predictions']])
    ax = plt.gca()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    plt.legend(['Train', 'Valid', 'Real-Test', 'Predictions'], loc='lower right')
    plt.grid()
    plt.savefig('./pictures/result_valid_test_3.jpg')

def residual_distribution():
    predictions, _ = test()
    # calculate residuals
    residuals = [test_data_label[i]-predictions[i] for i in range(len(predictions))]
    residuals = pd.DataFrame(residuals)
    print(residuals.describe())
    # residuals.plot()
    residuals.hist() # histogram plot
    plt.savefig('./pictures/residual_hist.jpg')
    residuals.plot(kind='kde') # density plot
    autocorrelation_plot(residuals)
    plt.savefig('./pictures/residual_autocorr.jpg')

if __name__ == "__main__":
    # train()
    result_plot()
    # residual_distribution()