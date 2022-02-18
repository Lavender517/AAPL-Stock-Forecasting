# Introduction

This project collects and stores **Apple Stock Price** from a specific period, as well as **Consumer Price Index (CPI)** and **Consumer Sentiment** data at the same time. After data preprocessing and data exploration, this report builds **Long Short-Term Memory (LSTM)** architecture both for univariate and multivariate predictions on the daily closing price of AAPL stock. The comparisons between LSTM and **Autoregressive Integrated Moving Average (ARIMA)** statistical model reveal the out-performance of the former, and the forecasting results show a significant correlation between Apple stock prices and auxiliary data above.


## File Directory
```

final-assignment-Lavender517
├── README.md
├── environment.yml
├── requirements.txt
├── /data/
├── /pictures/
├── /rnn_model/
├── /runs/
├── main.py
├── assignment.ipynb
├── assignment.py
├── untracked_api_key.txt
├── data_acquisition_storage.py
├── data_storage_database.py
├── data_processing_stock.py
├── data_processing_auxiliary.py
├── data_exploration.py
├── train_stock.py
├── train_stock_valid.py
├── train_auxiliary.py
├── train_auxiliary_upgrade.py
├── ARIMA.py

```

## File Description

```
environment.yml
requirements.txt
```
Provide specific list of external libraries and packages, the virtual environment builds on Anaconda.
```
data
```
Store the Apple stock prices and auxiliary data in various formats, including *.csv*, *.npy* and *json* files.
```
pictures
```
This folder saves the image outputs on each of the data analysis stage, such as data visualization and data exploration, etc.
```
rnn_model
```
Save the *pkl* files of each specific model that trained and produced in different employments in experiment.
```
runs
```
This folder saves the learing curves of each LSTM training process, they can be viewed by *TensorBoard* and draw the training loss of each experiment.
```
main.py
```
Integrate the functions that used to produce the data analysis process as well as model prediction results, which is used in *main* function in **assignment.py**.
```
untracked_api_key.txt
```
Store the api key which used in data acquisition process to connect to Alpha Vantage API.
```
data_storage_database.py
```
Implement the could-based data storage to interact with **MongoDB** to store and load data with it.
```
train_stock.py
```
Accomplish the training, validation, test and evaluation processes of LSTM model on univariate deployment.
```
train_stock_valid.py
```
Accomplish the training, validation and evaluation processes of LSTM model on univariate deployment.
```
train_auxiliary.py
```
Accomplish the training, validation, test and evaluation processes of LSTM model on multivariate deployment.
```
train_auxiliary_upgrade.py
```
Accomplish the training, validation, test and evaluation processes of LSTM model with multi-output on multivariate deployment.
```
ARIMA.py
```
Build appropriate model based on the data we collected, and compare its prediction results with LSTM to verify the effectiveness of our model.


## Execution

You just need to execute the *main* function in the assignment.py file to get all the outputs, like
```
python assignment.py
```