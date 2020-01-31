#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from datetime import date
from nsepy.history import get_price_list
from nsepy import get_history
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, neighbors
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc #pip install https://github.com/matplotlib/mpl_finance/archive/master.zip
from matplotlib.dates import date2num

def find_trend(stock):
	his = hist(stock)
	his = his.reset_index()
	# ind = his.index
	# ind = ind[5:]
	gain = his['Gain']
	loss = his['Loss']
	rsi = []
	for i in range(len(his)):
		g = []
		l = []
		g = gain[-5:]
		l = loss[-5:]
		avg_gain = sum(g)/5.0
		avg_loss = sum(l)/5.0
		if avg_loss!=0:
			rs = avg_gain/avg_loss
		else:
			rs = 0.0
		r = 100 - (100/(1+rs))
		gain = gain[:-1]
		loss = loss[:-1]
		rsi.append(r)
	#print(rsi) #['Date', 'Symbol', 'Prev Close', 'Open', 'High', 'Low', 'Last', 'Close', 'Change', 'Gain', 'Loss']
	his['RSI'] = rsi[::-1]
	#print(his)
	return his
##	ax1 = plt.subplot()
##	d = [int(str(his['Date'][i]).split('-')[-1]) for i in range(len(his['Date']))]
##	candlestick_ohlc(ax1,zip(his['Date'].apply(date2num),his['Open'],his['High'],his['Low'],his['Close']))
##	ax2 = plt.subplot()
##	ax2.plot(his['Date'],his['RSI'])
##	plt.show()

def gain_and_loss(data):
	gain = []
	loss = []
	for i in range(len(data)):
		if data['Change'][i] > 1:
			gain.append(data['Change'][i])
		else:
			gain.append(0)
	for i in range(len(data)):
		if data['Change'][i] < 1:
			loss.append(abs(data['Change'][i]))
		else:
			loss.append(0)
	return gain, loss

def hist(stock):
	s = get_history(symbol=stock, start=date(2018,12,1), end=date(2019,10,28))
	s.drop(['Series'], 1, inplace=True)
	s.drop(['VWAP'], 1, inplace=True)
	s.drop(['Volume'], 1, inplace=True)
	s.drop(['Turnover'], 1, inplace=True)
	s.drop(['Trades'], 1, inplace=True)
	s.drop(['Deliverable Volume'], 1, inplace=True)
	s.drop(['%Deliverble'], 1, inplace=True)
	s['Change'] = s['Close'] - s['Prev Close']
	s['Gain'], s['Loss'] = gain_and_loss(s)
	return s

def rsi_of_stock(stock):
	h = hist(stock)
	gain = h['Gain'][-5:]
	loss = h['Loss'][-5:]
	avg_gain = sum(gain)/5.0
	avg_loss = sum(loss)/5.0
	if avg_loss!=0:
		rs = avg_gain/avg_loss
	else:
		rs = 0.0
	rsi = 100 - (100/(1+rs))
	return rsi

def select_stock(data):
	for i in range(len(data)):
		try:
			if data['CLOSE'].loc[i] <= 50.0:
				data.drop(i, axis=0, inplace = True)
			elif data['CLOSE'].loc[i] >= 10000.0:
				data.drop(i, axis=0, inplace = True)
		except:
			continue
	return data

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data = get_price_list(dt=date(2019,1,4))
data.drop(['SERIES'], 1, inplace=True)
data.drop(['LAST'], 1, inplace=True)
data.drop(['TOTTRDQTY'], 1, inplace=True)
data.drop(['TOTTRDVAL'], 1, inplace=True)
data.drop(['TIMESTAMP'], 1, inplace=True)
data.drop(['TOTALTRADES'], 1, inplace=True)
data.drop(['ISIN'], 1, inplace=True)

data = select_stock(data)
rsi = []
predicts = []
# for i in range(len(data)):
data = data[:50]
for i in range(50):
	try:
		rsi.append(rsi_of_stock(data['SYMBOL'][i]))
	except:
		rsi.append(0)
	# print(i)
data = data.reset_index(drop=True)
data['RSI'] = rsi
for i in range(len(data)):
	if data['RSI'].loc[i] == 0.0:
		data.drop(i, axis=0, inplace = True)
data = data.reset_index(drop=True)
# for i in range(len(data)):
# 	predicts.append(find_trend(data['SYMBOL'][i]))
# data['TREND'] = predicts
#t = find_trend(data['SYMBOL'][1])
da = data.RSI >= 70
print("RSI >= 70\n", data[da])
fnames1 = data[da].SYMBOL
da = data.RSI <= 30
print("\n\nRSI <= 30\n", data[da])
# data.to_csv('Stock_data.csv', sep=',',index = False, encoding='utf-8')
fnames2 = data[da].SYMBOL

fnames = []
for r in fnames1:
    fnames.append(r)
for r in fnames2:
    fnames.append(r)
#print("\n\n", fnames)

new_data = find_trend(fnames[0]) #ACCELYA
new_data.drop('Symbol', axis=1, inplace=True)

#creating dataframe
data = new_data.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(new_data)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]
    
#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#print(new_data)

#creating train and test sets
dataset = new_data.values

train = dataset[0:300,:]
valid = dataset[60:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60, len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=5, batch_size=1, verbose=2)

#predicting
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
print(rms)

#for plotting
train = new_data[:300]
valid = new_data[60:]
valid['Predictions'] = closing_price
plt.plot(train['Close'][:65])
plt.plot(valid[['Predictions']])
plt.show()
