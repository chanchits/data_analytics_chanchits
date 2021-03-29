### Libraries ###
import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import pmdarima as pmd
from datetime import datetime
from matplotlib import rcParams
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from pandas_datareader.data import DataReader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, LSTM

import warnings
warnings.filterwarnings('ignore')


### Setting ###
colors = ['#4E79A9','#C70039','#A93226','#F28E2B','#59A14F','#BAB0AC','#E15759']
sns.set_palette(colors)
sns.set_style('whitegrid')
pd.set_option('max_columns', 30)

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
BIGGEST_SIZE = 18

rcParams['font.family'] = 'Times New Roman'
rcParams['font.weight'] = 'normal'
rcParams['figure.titleweight'] = 'bold'

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGEST_SIZE)  # fontsize of the figure title

os.getcwd()
os.chdir('.../path')


### BIG QUESTIONS TO ASK ###
"""
1) What was the change in price of the stock over time?
2) What was the daily return of the stock on average?
3) What was the MA of the various stocks?
4) What was the correlation between different stocks (In the same industry)?
5) How much value do we put at risk by investing in a particular stock?
6) How can we attempt to PREDICT future stock behavior? (ARIMA might not work, so Predicting the closing price stock price using LSTM (Long Short-Term Memory))
"""

### Import the dataset ###
start = '2018-01-01'
end = '2018-12-31'

kbank = pdr.get_data_yahoo('KBANK.BK', start=start, end=end)
bbl = pdr.get_data_yahoo('BBL.BK', start=start, end=end)
scb = pdr.get_data_yahoo('SCB.BK', start=start, end=end)
tmb = pdr.get_data_yahoo('TMB.BK', start=start, end=end)

kbank['Bank_Name'] = 'KBANK'
bbl['Bank_Name'] = 'BBL'
scb['Bank_Name'] = 'SCB'
tmb['Bank_Name'] = 'TMB'

bank_list = [kbank,bbl,scb,tmb]
df = pd.concat(bank_list)

fig, axes = plt.subplots(2,2, figsize=(16,8))
axes[0, 0].set_title('KBANK')
axes[0, 0].plot(df['Close'][df['Bank_Name']=='KBANK'], color='#59A14F')
axes[0, 1].set_title('BBL')
axes[0, 1].plot(df['Close'][df['Bank_Name']=='BBL'], color='#4E79A7')
axes[1, 0].set_title('SCB')
axes[1, 0].plot(df['Close'][df['Bank_Name']=='SCB'], color='purple')
axes[1, 1].set_title('TMB')
axes[1, 1].plot(df['Close'][df['Bank_Name']=='TMB'], color='#F28E2B')
plt.suptitle('Closing Price')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2,2, figsize=(16,8))
axes[0, 0].set_title('KBANK')
axes[0, 0].plot(df['Volume'][df['Bank_Name']=='KBANK'], color='#59A14F')
axes[0, 1].set_title('BBL')
axes[0, 1].plot(df['Volume'][df['Bank_Name']=='BBL'], color='#4E79A7')
axes[1, 0].set_title('SCB')
axes[1, 0].plot(df['Volume'][df['Bank_Name']=='SCB'], color='purple')
axes[1, 1].set_title('TMB')
axes[1, 1].plot(df['Volume'][df['Bank_Name']=='TMB'], color='#F28E2B')
plt.suptitle('Volume')
plt.tight_layout()
plt.show()

### Moving Average ###
kbank['MA_10_Adj_Close'] = kbank['Adj Close'].rolling(10).mean()
kbank['MA_20_Adj_Close'] = kbank['Adj Close'].rolling(20).mean()
kbank['MA_50_Adj_Close'] = kbank['Adj Close'].rolling(50).mean()

bbl['MA_10_Adj_Close'] = bbl['Adj Close'].rolling(10).mean()
bbl['MA_20_Adj_Close'] = bbl['Adj Close'].rolling(20).mean()
bbl['MA_50_Adj_Close'] = bbl['Adj Close'].rolling(50).mean()

scb['MA_10_Adj_Close'] = scb['Adj Close'].rolling(10).mean()
scb['MA_20_Adj_Close'] = scb['Adj Close'].rolling(20).mean()
scb['MA_50_Adj_Close'] = scb['Adj Close'].rolling(50).mean()

tmb['MA_10_Adj_Close'] = tmb['Adj Close'].rolling(10).mean()
tmb['MA_20_Adj_Close'] = tmb['Adj Close'].rolling(20).mean()
tmb['MA_50_Adj_Close'] = tmb['Adj Close'].rolling(50).mean()

fig, axes = plt.subplots(2,2, figsize=(16,8))
axes[0, 0].set_title('KBANK')
axes[0, 0].plot(kbank['Adj Close'], color='#59A14F')
axes[0, 0].plot(kbank['MA_10_Adj_Close'], color='#BEBEBE')
axes[0, 0].plot(kbank['MA_20_Adj_Close'], color='#808080')
axes[0, 0].plot(kbank['MA_50_Adj_Close'], color='#505050')
axes[0, 1].set_title('BBL')
axes[0, 1].plot(bbl['Adj Close'], color='#4E79A7')
axes[0, 1].plot(bbl['MA_10_Adj_Close'], color='#BEBEBE')
axes[0, 1].plot(bbl['MA_20_Adj_Close'], color='#808080')
axes[0, 1].plot(bbl['MA_50_Adj_Close'], color='#505050')
axes[1, 0].set_title('SCB')
axes[1, 0].plot(scb['Adj Close'], color='purple')
axes[1, 0].plot(scb['MA_10_Adj_Close'], color='#BEBEBE')
axes[1, 0].plot(scb['MA_20_Adj_Close'], color='#808080')
axes[1, 0].plot(scb['MA_50_Adj_Close'], color='#505050')
axes[1, 1].set_title('TMB')
axes[1, 1].plot(tmb['Adj Close'], color='#F28E2B')
axes[1, 1].plot(tmb['MA_10_Adj_Close'], color='#BEBEBE')
axes[1, 1].plot(tmb['MA_20_Adj_Close'], color='#808080')
axes[1, 1].plot(tmb['MA_50_Adj_Close'], color='#505050')
plt.suptitle('Adj Close & MA 10, 20, 50')
plt.legend(['Adj Close','MA_10_Adj_Close','MA_20_Adj_Close','MA_50_Adj_Close'], loc='upper right')
plt.show()

### Percentage Change ###
kbank['Daily_Return'] = kbank['Adj Close'].pct_change()
bbl['Daily_Return'] = bbl['Adj Close'].pct_change()
scb['Daily_Return'] = scb['Adj Close'].pct_change()
tmb['Daily_Return'] = tmb['Adj Close'].pct_change()

fig, axes = plt.subplots(2,2, figsize=(16,9))
axes[0, 0].set_title('KBANK')
axes[0, 0].plot(kbank['Daily_Return'], linestyle=':', marker='o',markersize=3, color='#59A14F')
axes[0, 1].set_title('BBL')
axes[0, 1].plot(bbl['Daily_Return'], linestyle=':', marker='o',markersize=3, color='#4E79A7')
axes[1, 0].set_title('SCB')
axes[1, 0].plot(scb['Daily_Return'], linestyle=':', marker='o',markersize=3, color='purple')
axes[1, 1].set_title('TMB')
axes[1, 1].plot(tmb['Daily_Return'], linestyle=':', marker='o',markersize=3, color='#F28E2B')
plt.suptitle('Percentage Change')
plt.show()


fig, axes = plt.subplots(2,2, figsize=(16,9))
axes[0, 0].set_title('KBANK')
sns.histplot(kbank['Daily_Return'].dropna(), bins=30, color='#59A14F', kde=True, ax=axes[0, 0])
axes[0, 1].set_title('BBL')
sns.histplot(bbl['Daily_Return'].dropna(), bins=30, color='#4E79A7', kde=True, ax=axes[0, 1])
axes[1, 0].set_title('SCB')
sns.histplot(scb['Daily_Return'].dropna(), bins=30, color='purple', kde=True, ax=axes[1, 0])
axes[1, 1].set_title('TMB')
sns.histplot(tmb['Daily_Return'].dropna(), bins=30, color='#F28E2B', kde=True, ax=axes[1, 1])
plt.suptitle('Histograms of Percentage Change')
plt.show()

#Note: Pct_change can tell us about the risks of each stock. However, most of them are similar. (Sideway)


### Correlation ###
bank_list = ['KBANK.BK','BBL.BK','SCB.BK','TMB.BK']
closing_df = DataReader(bank_list, 'yahoo', start, end)['Adj Close']
closing_df

bank_returns_df = closing_df.pct_change()
bank_returns_df

# Pairplot
sns.pairplot(bank_returns_df, kind='reg')
plt.show()

#Note: The pairplot shows that most of them are related, AKA, they go along together.

# Correlation Plot
fig, axes = plt.subplots(2,1, figsize=(16,8))
sns.heatmap(closing_df.corr(), annot=True, cmap='summer', ax=axes[0])
axes[0].set_title('Correlation of Percentage Return')
sns.heatmap(bank_returns_df.corr(), annot=True, cmap='summer', ax=axes[1])
axes[1].set_title('Correlation of Closing Price (Adj Close)')
plt.suptitle('Correlation')
plt.show()

#Note: BBL & TMB seem to have no correlation according to Percentage Return
#Note: KBANK & SCB seem to have strong correlation according to Percentage Return



### Risk and Return ###
# There are many ways we can quantify risk, 
# one of the most basic ways using the information we've gathered on 
# daily percentage returns is by comparing the expected return with the 
# standard deviation of the daily returns.

returns = bank_returns_df.dropna()
returns

area = np.pi*40
plt.figure(figsize=(16,8))
plt.scatter(returns.mean(), returns.std(), s=area)
plt.xlabel('Expected Return')
plt.ylabel('Risk')
for label, x, y in zip(returns.columns, returns.mean(), returns.std()):
    plt.annotate(label, xy=(x, y), xytext=(50, 50), textcoords='offset points', ha='right', va='bottom', 
                arrowprops=dict(arrowstyle='-', color='#4E79A7', connectionstyle='arc3,rad=-0.3'))
plt.show()

#Well, this seems BBL has the highest return with lowest risk.


### Predicting the closing price of KBANK using Long Short-Term Memory (LSTM) ###
start = '2018-01-01'
end = '2018-12-31'

kbank_df = pdr.get_data_yahoo('KBANK.BK', start=start, end=end)
kbank_df

plt.figure(figsize=(16,9))
plt.title('Close Price History')
plt.plot(kbank_df['Close'])
plt.xlabel('Date')
plt.ylabel('Close Price THB')
plt.show()

data = kbank_df.filter(['Close'])
dataset = data.values
training_data_len = int(np.ceil(len(dataset) * 0.8)) #Training:Test = 80:20
training_data_len

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data


train_data = scaled_data[0:int(training_data_len), :]

#Split the dataset
#Training Set
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 61:
        print(x_train)
        print(y_train)
        print()


x_train, y_train = np.array(x_train), np.array(y_train) # Convert the x_train and y_train to numpy arrays
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # Reshape the data

#Test Set
test_data = scaled_data[training_data_len - 60: , :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))



# Model Creation
model = Sequential()
model.add(LSTM(136, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)


### PREDICTIONS ###
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
rmse

# Prediction and Visualization

train = data[:training_data_len]
test = data[training_data_len:]
test = data[training_data_len:]
test['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('KBANK MODEL')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price THB', fontsize=18)
plt.plot(train['Close'])
plt.plot(test[['Close', 'Predictions']])
plt.legend(['Train', 'Test (Actual)', 'Predictions'], loc='upper right')
plt.show()

test.head()

### Conclusion ###
"""
Well, the prediction result seems to go along with the trend, however, the accuracy might not be satisfied.
Another model should be tried is 'ARIMA' with seasonal. (Could be fitter than LSTM)
"""
