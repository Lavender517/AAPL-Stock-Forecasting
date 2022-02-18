#!/usr/bin/env python
# coding: utf-8

# # Data Processing: Auxiliary Data
# **CPI & Consumer Sentiment**
# ### Import Libraries & Load Data

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns


# In[2]:


# apple_raw = pd.read_csv('./data/apple_raw.csv')
# apple_raw['Date'] = pd.to_datetime(apple_raw['Date'])
# apple_raw.set_index("Date", inplace=True)

# apple_test = pd.read_csv('./data/apple_raw_test.csv')
# apple_test['Date'] = pd.to_datetime(apple_test['Date'])
# apple_test.set_index("Date", inplace=True)


# ## 1. CPI (Consumer Price Index)
# ### 1.1 Data Resample

# In[3]:

def cpi_resample_all(apple_raw, apple_test, df_cpi_train, df_cpi_test):
    def cpi_resample(df):
        cpi_raw = df
        cpi_raw['Date'] = pd.to_datetime(cpi_raw['Date'])
        cpi_raw.set_index("Date", inplace=True)
        cpi_raw = cpi_raw.resample('D').interpolate() # Up samping to every day
        return cpi_raw

    df_cpi_raw = pd.read_csv('./data/data_CPI.csv')
    cpi_raw = cpi_resample(df_cpi_raw)
    all_cpi = apple_raw.join(cpi_raw) # Merge time series stock data and CPI data
    cpi = all_cpi['CPI'] # Extract CPI data

    df_cpi_test = pd.read_csv('./data/data_CPI_test.csv')
    cpi_test = cpi_resample(df_cpi_test)
    all_test_cpi = apple_test.join(cpi_test)
    return all_cpi, cpi, all_test_cpi


# In[4]:


# df_cpi_test = pd.read_csv('./data/data_CPI_test.csv')
# cpi_test = cpi_resample(df_cpi_test)
# all_test_cpi = apple_test.join(cpi_test)


# In[20]:


# cpi.describe().T


# ### 1.2 Data Clean

# In[21]:

def check_missing_value_cpi(cpi):
    if cpi.isna().mean() == 0:
        print("There is no null value in the CPI data")

# cpi.isna().mean()


# In[7]:

def check_outliers_cpi(cpi):
    # Using Z-score to find outliers
    from scipy import stats
    z = np.abs(stats.zscore(cpi))

    print('z score of the dataset is:\r\n',z)

    plt.figure(figsize=(8, 5))
    plt.plot(cpi.index, z)
    plt.xlabel('Date')
    plt.ylabel('CPI Z score')
    plt.grid()
    plt.title('Z-score Outlier Test for CPI')
    plt.savefig('./pictures/Zscore_CPI.jpg')

    # set a threshold and find the location where the value meets our condition(s)
    threshold = 3
    outlier_loc = np.where(z > threshold)
    if len(outlier_loc) > 1:
        # find the outlier value given its index
        outlier_by_Z_Score = cpi.values[outlier_loc]
        print('the data classified as outlier by z score:\r\n', outlier_by_Z_Score)
        print('the date of the outlier is:\r\n', cpi.index[outlier_loc])
    else:
        print("There is no outliers in the time series stock data")


# ### 1.3 Data Visualization

# In[29]:

def visualization_cpi(cpi, all_cpi, df_cpi_raw, apple_raw):
    cpi.plot(figsize=(16, 7))
    plt.xlabel('Date')
    plt.ylabel('CPI')
    plt.title('CPI from April 2017 to April 2021')
    plt.grid()
    plt.savefig('./pictures/CPI_overall.jpg')


# In[132]:


    start, end = '2017-04-01', '2021-04-01' 
    fig, ax = plt.subplots(figsize=(6, 5)) 

    ax.plot(all_cpi.loc[start:end, 'CPI'], marker='.', linestyle='-', linewidth = 0.5, label='Daily', color='black') 
    ax.plot(df_cpi_raw.loc[start:end, 'CPI'], marker='o', markersize=8, linestyle='-', label='Weekly', color='coral')
    ax.set_ylabel("CPI") 
    ax.legend()
    plt.savefig('./pictures/CPI_resample.jpg')


# In[209]:


    import calplot


    calplot.calplot(all_cpi['CPI'], vmin=230, vmax=280, edgecolor=None, cmap='BuPu', 
                    suptitle='CPI Daily Values', textfiller='-')
    plt.savefig('./pictures/CPI_heatmap.jpg')


# In[22]:


    # sns.set(rc = {'figure.figsize':(10, 3)})
    # plt.figure(figsize = (15,8))
    sns.jointplot(x=cpi, y=apple_raw['Adj Close'], kind='reg', color='seagreen', height=7)
    plt.gcf().set_size_inches(6, 5)
    plt.xlabel('Date')
    plt.ylabel('CPI')
    # plt.title('CPI and stock price correlation')
    plt.grid()
    plt.savefig('./pictures/CPI_correlation.jpg')


# ### 1.4 Data Transformation
# Do the normailzation of output data, using Min-Max scaling in sklearn to implement it.

# In[99]:

def transformation_cpi(all_cpi):
    cpi_col = all_cpi.filter(['CPI'])
    # Convert the dataframe to a numpy array
    cpi_val = cpi_col.values

    # Scale the data
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_cpi = scaler.fit_transform(cpi_val)
    print("The scaled data are:\r\n", scaled_cpi)
    return scaled_cpi


# ## 2. Consumer Sentiment
# ### 2.1 Data Resample

# In[148]:

def sentiment_resample_all(df_sent_train, df_sent_test_, all_cpi, all_test_cpi):
    def sentiment_resample(df):
        sentiment_raw = df
        sentiment_raw['Date'] = pd.to_datetime(sentiment_raw['Date'])
        sentiment_raw.set_index("Date", inplace=True)
        sentiment_raw = sentiment_raw.resample('D').interpolate() # Up samping to every day
        return sentiment_raw

    # df_sent_raw = pd.read_csv('./data/data_sentiment.csv', parse_dates=True, index_col = "Date")
    df_sent_raw = pd.read_csv('./data/data_sentiment.csv')
    sentiment_raw = sentiment_resample(df_sent_raw)
    all_sentiment = all_cpi.join(sentiment_raw) # Merge time series stock data and CPI data
    all_sentiment.to_csv("./data/data_all.csv", date_format = "%Y-%m-%d", index = True)
    sentiment = all_sentiment['Sentiment'] # Extract CPI data

    df_sent_test = pd.read_csv('./data/data_sentiment_test.csv')
    sent_test = sentiment_resample(df_sent_test)
    all_test = all_test_cpi.join(sent_test)
    return sentiment, all_sentiment, all_test


# In[9]:


# df_sent_test = pd.read_csv('./data/data_sentiment_test.csv')
# sent_test = sentiment_resample(df_sent_test)
# all_test = all_test_cpi.join(sent_test)


# In[63]:


# sentiment.describe()


# ### 2.2 Data Clean

# In[64]:

def check_missing_value_sent(sentiment):
    if sentiment.isna().mean() == 0:
        print("There is no null value in the Sentiment data")



# In[10]:

def check_outliers_sent(sentiment):
    # Using Z-score to find outliers
    from scipy import stats
    z = np.abs(stats.zscore(sentiment))

    print('z score of the dataset is:\r\n',z)

    plt.figure(figsize=(8, 5))
    plt.plot(sentiment.index, z)
    plt.xlabel('Date')
    plt.ylabel('Sentiment Z score')
    plt.grid()
    plt.title('Z-score Outlier Test for Consumer Sentiment')
    plt.savefig('./pictures/Zscore_sentiment.jpg')

    # set a threshold and find the location where the value meets our condition(s)
    threshold = 3
    outlier_loc = np.where(z > threshold)
    if len(outlier_loc) > 1:
        # find the outlier value given its index
        outlier_by_Z_Score = sentiment.values[outlier_loc]
        print('the data classified as outlier by z score:\r\n', outlier_by_Z_Score)
        print('the date of the outlier is:\r\n', sentiment.index[outlier_loc])
    else:
        print("There is no outliers in consumer sentiement data")


# ### 2.3 Data Visualization

# In[68]:

def visualization_sent(sentiment, all_sentiment, df_sent_raw, apple_raw):
    sentiment.plot(figsize=(16, 7), c='g')
    plt.xlabel('Date')
    plt.ylabel('Sentiment')
    plt.title('Consumer Sentiment from April 2017 to April 2021')
    plt.grid()
    plt.savefig('./pictures/Sentiment_overall.jpg')


# In[133]:


    start, end = '2017-04-01', '2021-04-01' 
    fig, ax = plt.subplots(figsize=(6, 5)) 
    
    all_sentiment['Sentiment']

    ax.plot(all_sentiment.loc[start:end, 'Sentiment'], marker='.', linestyle='-', linewidth = 0.5, label='Daily', color='black') 
    ax.plot(df_sent_raw.loc[start:end, 'Sentiment'], marker='o', markersize=8, linestyle='-', label='Weekly', color='coral')
    ax.set_ylabel("Sentiment") 
    ax.legend()
    plt.savefig('./pictures/Sentiment_resample.jpg')


# In[149]:


# all_sentiment


# In[211]:


    import calplot

    calplot.calplot(all_sentiment['Sentiment'], vmin=80, vmax=110, edgecolor=None, cmap='GnBu',
                    suptitle='Sentiment Daily Values', textfiller='-')
    plt.savefig('./pictures/Sentiment_heatmap.jpg')


# In[21]:


    sns.jointplot(x=sentiment, y=apple_raw['Adj Close'], kind='reg', color='blue', height=7)
    plt.gcf().set_size_inches(6, 5)
    plt.xlabel('Date')
    plt.ylabel('Sentiment')
    # plt.title('Consumer sentiment and stock price correlation')
    plt.grid()
    plt.savefig('./pictures/sentiment_correlation.jpg')


# ### 2.4 Data Transformation
# Do the normailzation of output data, using Min-Max scaling in sklearn to implement it.

# In[93]:

def transformation_sent(all_sentiment):
    sentiment_col = all_sentiment.filter(['Sentiment'])
    # Convert the dataframe to a numpy array
    sentiment_val = sentiment_col.values


    # Scale the data
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_sentiment = scaler.fit_transform(sentiment_val)
    print("The scaled data are:\r\n", scaled_sentiment)
    return scaled_sentiment


# In[104]:


# # Create a new dataframe with only the 'Adj Close column'
# adj_close = apple_raw.filter(['Adj Close'])
# # Convert the dataframe to a numpy array
# stock = adj_close.values

# # Scale the data
# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler(feature_range=(0,1))
# scaled_data = scaler.fit_transform(stock)
# print("The scaled data are:\r\n", scaled_data)


# In[105]:


def draw_normalization_figure(df_train, scaled_data, scaled_cpi, scaled_sentiment):
    y = [scaled_data, scaled_cpi, scaled_sentiment]
    scaled_data = np.swapaxes(scaled_data, 0, 1)
    scaled_data = scaled_data.flatten()

    scaled_cpi = np.swapaxes(scaled_cpi, 0, 1)
    scaled_cpi = scaled_cpi.flatten()

    scaled_sentiment = np.swapaxes(scaled_sentiment, 0, 1)
    scaled_sentiment = scaled_sentiment.flatten()


# In[127]:


    # pal = sns.color_palette("Set2")
    plt.figure(figsize=(15, 5))
    plt.stackplot(df_train.index.values, scaled_data, scaled_cpi, scaled_sentiment,
                labels=['AAPL Stock Price','CPI','Consumer Sentiment'])
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend(loc='upper left')
    plt.savefig('./pictures/normalization.jpg')


# ## 3. Data Integration

# In[6]:

def data_integration(all_sentiment, all_test):
    all_sentiment.to_csv("./data/data_all.csv", date_format = "%Y-%m-%d", index = True)



# In[21]:


    all_test.to_csv("./data/data_all_test.csv", date_format = "%Y-%m-%d", index = True)


# In[7]:


# all_sentiment.describe().T


# In[ ]:




