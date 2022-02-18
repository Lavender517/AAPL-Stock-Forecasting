#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing: Time Series Stock Data
# 
# ### Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns


# ### Load Times Series Apple Stock Data

# In[2]:


apple_raw = pd.read_csv('./data/apple_raw.csv', index_col=0)


# ### Describe the data

# In[3]:


apple_raw.describe().T


# ## 1. Data Clean
# ### 1.1 Check whether missing values exist

# In[4]:

def check_missing_value(apple_raw):
    if sum(apple_raw.isna().mean()) == 0:
        print("There is no null value in the time series stock data")


# ### 1.2 Check whether outliers exist

# In[9]:

def check_outliers(apple_raw):
    # Using Z-score to find outliers
    from scipy import stats
    z = np.abs(stats.zscore(apple_raw['Adj Close']))

    print('z score of the dataset is:\r\n',z)

    totalSeed = apple_raw.index.tolist()
    xticks=list(range(0,len(totalSeed), 95))
    xlabels=[totalSeed[x] for x in xticks]
    xticks.append(len(totalSeed))
    xlabels.append(totalSeed[-1])

    plt.figure(figsize=(20, 5))
    plt.plot(apple_raw.index, z)
    ax = plt.gca()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    plt.xlabel('Date')
    plt.ylabel('Z score')
    plt.grid()
    plt.title('Z-score Outlier Test for AAPL Stock Prices')
    plt.savefig('./pictures/Zscore_raw_data.jpg')

    # set a threshold and find the location where the value meets our condition(s)
    threshold = 3
    outlier_loc = np.where(z > threshold)
    if len(outlier_loc) > 1:
        # find the outlier value given its index
        outlier_by_Z_Score = apple_raw['Adj Close'].values[outlier_loc]
        print('the data classified as outlier by z score:\r\n', outlier_by_Z_Score)
        print('the date of the outlier is:\r\n', apple_raw.index[outlier_loc])
    else:
        print("There is no outliers in the time series stock data")


# ## 2. Data Visulization
# ### 2.1 Overall Trend

# In[7]:

def visualization(df_train):
    apple_raw = pd.read_csv('./data/apple_raw.csv', index_col=0)
    apple_raw.plot(subplots=True, figsize=(20, 12))
    plt.savefig('./pictures/data_subplots.jpg')
    print('Draw and save subplots successfully')


# In[10]:


# Prepare data first
    price = apple_raw.drop(columns=['Volume']) # Data without volume
    volume = apple_raw['Volume'] # Volume data


# ### 2.2 Analysis on Price and Volume from 2017.4 to 2021.4

# In[9]:

    totalSeed = apple_raw.index.tolist()
    xticks=list(range(0,len(totalSeed), 95))
    xlabels=[totalSeed[x] for x in xticks]
    xticks.append(len(totalSeed))
    xlabels.append(totalSeed[-1])

    plt.figure(figsize=(16, 7))
    plt.plot(price)
    ax = plt.gca()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Apple Stocks from April 2017 to April 2021')
    plt.grid()
    plt.savefig('./pictures/stock_price.jpg')
    print('Draw and save stock prices analysis successfully')


# In[10]:


    plt.figure(figsize=(16, 7))
    plt.plot(volume)
    ax = plt.gca()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.title('Apple Volume Stock from April 2017 to April 2021')
    plt.grid()
    plt.savefig('./pictures/stock_volume.jpg')
    print('Draw and save stock volume analysis successfully')


# ### 2.3 Area Plots on High, Low, Open, Close and Adj Close from 2017.4 to 2021.4

# In[11]:


    fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True, figsize=(20, 10))
    ax1.set_title('High and Low')
    price['High'].plot(kind='area', ax=ax1)
    price['Low'].plot(kind='area', ax=ax1)
    ax1.set_xlabel('Years')
    ax1.grid()

    ax2.set_title('Close and Adj Close')
    price['Close'].plot(kind='area', ax=ax2, color='g')
    price['Adj Close'].plot(kind='area', ax=ax2, color='y')
    ax2.set_xlabel('Years')
    ax2.grid()

    plt.savefig('./pictures/area_plot.jpg')
    print('Draw and save area plot successfully')


# ### 2.4 Time Series Lag Plot
# A lag plot helps to check if a time series data set is random or not. A random data will be evenly spread whereas a shape or trend indicates the data is not random.

# In[16]:


    plt.figure(figsize=(20,5))

    # # Stock High
    # plt.subplot(2, 2, 1)
    # pd.plotting.lag_plot(apple_raw['High'])
    # plt.title("High")

    # # Stock Open
    # plt.subplot(2, 2, 2)
    # pd.plotting.lag_plot(apple_raw['Open'])
    # plt.title("Open")

    # Stock Adj Close
    plt.subplot(1, 2, 1)
    pd.plotting.lag_plot(apple_raw['Adj Close'])
    plt.title("Adj Close")

    # Stock Volume
    plt.subplot(1, 2, 2)
    pd.plotting.lag_plot(apple_raw['Volume'])
    plt.title("Volume")

    # plt.suptitle("Time Series Lag Plot")
    plt.grid()
    plt.savefig('./pictures/lag_plot.jpg')
    print('Draw and save time series lag plot successfully')


# ## 3. Data Transformation
# Do the normailzation of output data, using Min-Max scaling in sklearn to implement it.

# In[5]:

def transformation(apple_raw):
    # Create a new dataframe with only the 'Adj Close column'
    adj_close = apple_raw.filter(['Adj Close'])
    # Convert the dataframe to a numpy array
    stock = adj_close.values

    # Scale the data
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(stock)
    print("The scaled data are:\r\n", scaled_data)
    return scaled_data
