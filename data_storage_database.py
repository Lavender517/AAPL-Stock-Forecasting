from optparse import IndentedHelpFormatter
import pymongo
from pymongo import MongoClient
import certifi
from pprint import pprint
from random import randint
import json
import requests 
import pandas as pd
from data_acquisition_storage import sent2csv, cpi2csv

# client = pymongo.MongoClient('localhost', 27017) # Connect from localhost

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

def load_json_stock(data_stock, db_data):
    for key in data_stock["Time Series (Daily)"].keys():
        everyday_data = data_stock["Time Series (Daily)"][key]
        json_data = {
            'Date': key,
            'Open': everyday_data["1. open"],
            'High': everyday_data["2. high"],
            'Low': everyday_data["3. low"],
            'Close': everyday_data["4. close"],
            'volume': everyday_data["5. volume"]
        }
        db_data.insert_one(json_data)
        print(json_data)

def load_json_aux(data, db_data, name=''):
    for item in data['data']:
        json_data = {
            'Date': item['date'],
            name: item['value']
        }
        # write data
        db_data.insert_one(json_data)
        print(json_data)

# load_json_stock(data_stock, db_apple_stock)
# print('Finished stock data loading')
# load_json_aux(data_sentiment, db_sentiment, name='Sentiment')
# print('Finished sentiment data loading')
# load_json_aux(data_cpi, db_cpi, name='CPI')
# print('Finished CPI data loading')

def acq_stock_db():
    stock_acq = db['Apple_Stock']
    stock_json = list(stock_acq.find())
    df = pd.DataFrame.from_dict(stock_json)
    df.reset_index(inplace=True)
    df = df.rename(columns={"1. open": "Open", "2. high": "High","3. low": "Low", "4. close": "Close", "5. volume": "Volume"})
    df = df.drop(['_id'], axis=1)
    df.index = df.index.astype('datetime64[ns]')
    df.to_csv("./data/apple_raw_db.csv",date_format="%Y-%m-%d",index = False)

def acq_aux_db(aux_name):
    aux_acq = db[aux_name]
    aux_json = list(aux_acq.find())
    if aux_name == 'Sentiment':
        sent2csv(aux_json)
    else:
        cpi2csv(aux_json)

if __name__ == "__main__":
    acq_stock_db()
    # acq_aux_db('Sentiment')
    # acq_aux_db('CPI')