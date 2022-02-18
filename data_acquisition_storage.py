from msilib import datasizemask
from unittest.case import DIFF_OMITTED
import pandas as pd
from pandas.io.json import json_normalize
import pandas_datareader as pdr
import numpy as np
from matplotlib import pyplot as plt
import json
import requests 
import csv
import os.path

'''
Data Acquisition from 3 types:
1. Apple's Stock prices over time period.
2. Monthly consumer sentiment and confidence data of the United States.
3. Monthly consumer price index (CPI) of the United States.

Data Time Period: From April 2017 to April 2021

Data Storage using 2 methods:
1. In cloud-based database: Mongo DB, Alpha Vantage API
2. In local files (Ex: csv file)
'''

#Load API key:
def get_api_key():
    if os.path.isfile("untracked_api_key.txt") == True:
        with open('untracked_api_key.txt', 'r') as file_object:
            #Text document that is untracked in directory (NOT pushed to git) with API key
            #Create a file called untracked_api_key.txt with your key and DO NOT add to git when commiting or pushing changes
            api_key = file_object.readline()
    elif os.path.isfile("untracked_api_key.txt") == False:
        print("untracked_api_key.txt not in directory; please provide API key in a file named untracked_api_key.txt in the same directory.")
    return api_key

# Choose Apple Inc.
stock_ticker = "AAPL"
api_key = get_api_key()

def get_stock_data():
    df = pdr.get_data_yahoo(stock_ticker)
    df_train = df['2017-04-01':'2021-04-30']
    df_test = df['2021-05-01':'2021-05-31']
    df_train.to_csv("./data/apple_raw.csv", date_format = "%Y-%m-%d", index = True)
    df_test.to_csv("./data/apple_raw_test.csv", date_format = "%Y-%m-%d", index = True)
    print("---Finished Time Series Stock Data Acquisition---")
    return df_train, df_test

def get_stock_json():
    target_url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol="+stock_ticker+"&outputsize=full&apikey="+api_key
    
    print("API request: " + target_url)
    apple_raw = requests.get(target_url).json()

	#Save JSON response to file
    with open('./data/apple_raw','w+') as outfile:
        json.dump(apple_raw, outfile, indent=4, sort_keys=True)

    return apple_raw

def get_sentiment_data():   
    #API request (monthly consumer sentiment data)
    target_url_sentiment = "https://www.alphavantage.co/query?function=CONSUMER_SENTIMENT&apikey=" + api_key
    print("API request: " + target_url_sentiment)
    
    data_sentiment = requests.get(target_url_sentiment).json()
    
    #Save JSON response to file
    with open('./data/data_sentiment', 'w+') as outfile:
        json.dump(data_sentiment, outfile, indent=4, sort_keys=True)

    return data_sentiment

def sent2csv(data_sentiment):
    #Save JSON response to CSV
    df_sentiment = pd.DataFrame.from_dict(data_sentiment)
    df_sentiment = df_sentiment.rename(columns = {"date": "Date", "value": "Sentiment"})
    df_sentiment = df_sentiment[['Date', 'Sentiment']]
    df_sentiment['Date'] = pd.to_datetime(df_sentiment['Date'])
    df_sentiment = df_sentiment.set_index('Date')
    df_sentiment.sort_index(ascending=True, inplace=True)
    df_sentiment_train = df_sentiment['2017-04-01':'2021-05-01']
    df_sentiment_test = df_sentiment['2021-05-01':'2021-06-01']
    df_sentiment_train.to_csv("./data/data_sentiment.csv", date_format = "%Y-%m-%d", index = True)
    df_sentiment_test.to_csv("./data/data_sentiment_test.csv", date_format = "%Y-%m-%d", index = True)
    return df_sentiment_train, df_sentiment_test

def get_sentiment():
    data_sentiment = get_sentiment_data()
    df_sentiment_train, df_sentiment_test = sent2csv(data_sentiment['data'])
    print("---Finished Consumer Sentiment Data Acquisition---")
    return data_sentiment, df_sentiment_train, df_sentiment_test

def get_cpi_data():
    #API request (monthly consumer sentiment data)
    target_url_sentiment = "https://www.alphavantage.co/query?function=CPI&interval=monthly&apikey=" + api_key
    print("API request: " + target_url_sentiment)
    
    data_CPI = requests.get(target_url_sentiment).json()
    
    #Save JSON response to file
    with open('./data/data_CPI', 'w+') as outfile:
        json.dump(data_CPI, outfile, indent=4, sort_keys=True)

    return data_CPI

def cpi2csv(data_CPI):
    #Save JSON response to CSV
    df_cpi = pd.DataFrame.from_dict(data_CPI)
    df_cpi = df_cpi.rename(columns = {"date": "Date", "value": "CPI"})
    df_cpi = df_cpi[['Date', 'CPI']]
    df_cpi['Date'] = pd.to_datetime(df_cpi['Date'])
    df_cpi = df_cpi.set_index('Date')
    df_cpi.sort_index(ascending=True, inplace=True)
    df_cpi_train = df_cpi['2017-04-01':'2021-05-01']
    df_cpi_test = df_cpi['2021-05-01':'2021-06-01']
    df_cpi_train.to_csv("./data/data_CPI.csv", date_format = "%Y-%m-%d", index = True)
    df_cpi_test.to_csv("./data/data_CPI_test.csv", date_format = "%Y-%m-%d", index = True)
    return df_cpi_train, df_cpi_test

def get_cpi():
    data_cpi = get_cpi_data()
    df_cpi_train, df_cpi_test = cpi2csv(data_cpi['data'])
    print("---Finished CPI Data Acquisition---")
    return data_cpi, df_cpi_train, df_cpi_test

if __name__ == "__main__":
    # Choose Apple Inc.
    stock_ticker = "AAPL"
    api_key = get_api_key()
    df_train, df_test = get_stock_data()
    data_sentiment, df_sentiment_train, df_sentiment_test = get_sentiment()
    data_cpi, df_cpi_train, df_cpi_test = get_cpi()