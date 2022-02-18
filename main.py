from data_acquisition_storage import get_api_key, get_stock_data, get_sentiment_data, sent2csv, get_cpi_data, cpi2csv
import pymongo
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data_storage_database import load_json_stock, load_json_aux
from data_processing_stock import check_missing_value, check_outliers, visualization, transformation
from data_processing_auxiliary import cpi_resample_all, check_missing_value_cpi, check_outliers_cpi, visualization_cpi, transformation_cpi
from  data_processing_auxiliary import sentiment_resample_all, check_missing_value_sent, check_outliers_sent, transformation_sent, visualization_sent
from data_processing_auxiliary import draw_normalization_figure, data_integration
from data_exploration import EDA, hypothesis_testing_sent, hypothesis_testing_cpi
from train_stock import getData, TrainSet, DataLoader, train, result_plot



def acquire(data):
    # Choose Apple Inc.
    stock_ticker = "AAPL"
    api_key = get_api_key()
    if data == 'CPI':
        data_cpi = get_cpi_data()
        return data_cpi
    elif data == 'sentiment':
        data_sentiment = get_sentiment_data()
        return data_sentiment
    else:
        df_train, df_test = get_stock_data()
        return df_train, df_test


def store(data_acquired, name):
    if name == 'CPI':
        df_cpi_train, df_cpi_test = cpi2csv(data_acquired['data'])
        print("---Finished CPI Data Storage---")
        return df_cpi_train, df_cpi_test
    elif name == 'sentiment':
        df_sentiment_train, df_sentiment_test = sent2csv(data_acquired['data'])
        print("---Finished Consumer Sentiment Data Acquisition---")
        return df_sentiment_train, df_sentiment_test



def retrieve(data):
    client = pymongo.MongoClient("mongodb+srv://riddles:1234@cluster0.aj9ti.mongodb.net/admin?retryWrites=true&w=majority", tlsCAFile=certifi.where())
    db = client['DAPS']
    db_apple_stock = db['Apple_Stock']
    db_sentiment = db['Sentiment']
    db_cpi = db['CPI']

    # stock_ticker = "AAPL"
    # api_key = get_api_key()
    with open('./data/apple_raw','r') as f:
        data_stock = json.load(f)
    with open('./data/data_sentiment','r') as f:
        data_sentiment = json.load(f)
    with open('./data/data_cpi','r') as f:
        data_cpi = json.load(f)

    load_json_stock(data_stock, db_apple_stock)
    print('Finished stock data loading')
    load_json_aux(data_sentiment, db_sentiment, name='Sentiment')
    print('Finished sentiment data loading')
    load_json_aux(data_cpi, db_cpi, name='CPI')
    print('Finished CPI data loading')

def describe(data):
    return data.describe().T

def process(df_train, df_test, df_cpi_train, df_cpi_test, df_sentiment_train, df_sentiment_test):
    # Stock Preprocessing
    
    check_missing_value(df_train)
    check_outliers(df_train)
    visualization(df_train)
    scaled_stock = transformation(df_train)

    # CPI Preprocessing
    all_cpi, cpi, all_test_cpi = cpi_resample_all(df_train, df_test, df_cpi_train, df_cpi_test)

    check_missing_value_cpi(cpi)
    check_outliers_cpi(cpi)
    visualization_cpi(cpi, all_cpi, df_cpi_train, df_train)
    print("---CPI Visualiation Finished---")
    scaled_cpi = transformation_cpi(all_cpi)

    # Sentiment Preprocessing
    sentiment, all_sentiment, all_test = sentiment_resample_all(df_sentiment_train, df_sentiment_test, all_cpi, all_test_cpi)

    check_missing_value_sent(sentiment)
    check_outliers_sent(sentiment)
    visualization_sent(sentiment, all_sentiment, df_sentiment_train, df_train)
    print("---Sentiment Visualiation Finished---")
    scaled_sent = transformation_sent(all_sentiment)

    # Normalization Figure
    draw_normalization_figure(df_train, scaled_stock, scaled_cpi, scaled_sent)

    # Data Integation
    data_integration(all_sentiment, all_test)

    # return scaled_stock, scaled_cpi, scaled_sent
    return all_sentiment, all_test


def explore(all_train, df_train):

    stock_daily_return = EDA(all_train, df_train)

    cpi_return, df_return = hypothesis_testing_sent(all_train, stock_daily_return)
    hypothesis_testing_cpi(stock_daily_return, cpi_return, df_return)

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


# from train_stock import *
def train_model(model, df_train):

    # Define Hyper-parameters
    LR = 0.0001
    EPOCH = 100
    DAYS_BEFORE = 30

    # Implement Trainig Dataset
    # df = pd.read_csv('./data/apple_raw.csv', index_col=0)
    train_data, train_series, df_index = getData(df_train, 'Adj Close', days_before=DAYS_BEFORE)

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
    # rnn = model

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.MSELoss() # Mean Squared Error

    train()
    result_plot()

def evaluate(val_data):
    from train_stock import residual_distribution
    
    residual_distribution()