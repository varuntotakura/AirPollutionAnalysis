import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import time
import datetime

plt.style.use("seaborn-colorblind")
data4 = pd.read_csv("C:/Users/VARUN/Desktop/AirPollution/Dataset/cpcb_dly_aq_andhra_pradesh-2004.csv")[['Sampling Date', 'City/Town/Village/Area', 'SO2', 'NO2']]
data6 = pd.read_csv("C:/Users/VARUN/Desktop/AirPollution/Dataset/cpcb_dly_aq_andhra_pradesh-2006.csv")[['Sampling Date', 'City/Town/Village/Area', 'SO2', 'NO2']]
data7 = pd.read_csv("C:/Users/VARUN/Desktop/AirPollution/Dataset/cpcb_dly_aq_andhra_pradesh-2007.csv")[['Sampling Date', 'City/Town/Village/Area', 'SO2', 'NO2']]
data9 = pd.read_csv("C:/Users/VARUN/Desktop/AirPollution/Dataset/cpcb_dly_aq_andhra_pradesh-2009.csv")[['Sampling Date', 'City/Town/Village/Area', 'SO2', 'NO2']]
data11 = pd.read_csv("C:/Users/VARUN/Desktop/AirPollution/Dataset/cpcb_dly_aq_andhra_pradesh-2011.csv")[['Sampling Date', 'City/Town/Village/Area', 'SO2', 'NO2']]
data14 = pd.read_csv("C:/Users/VARUN/Desktop/AirPollution/Dataset/cpcb_dly_aq_telangana-2014.csv")[['Sampling Date', 'City/Town/Village/Area', 'SO2', 'NO2']]
data = pd.concat([data4, data6, data7, data9, data11, data14], axis=0)
data = data.reset_index(drop=True)
#print(data.head())

for i in range(len(data)):
    if data['City/Town/Village/Area'][i] != 'Hyderabad':
        data.drop(i, inplace = True)

for i in range(len(data)):
    try:
        data['Sampling Date'][i] = time.mktime(datetime.datetime.strptime(data['Sampling Date'][i], "%d/%m/%Y").timetuple())
    except:
        pass
data = data[['Sampling Date', 'SO2', 'NO2']]
print(data.head())

no2 = data[['Sampling Date', 'NO2']]
forecast_len = 30
x=np.array(no2.drop(['NO2'],1))
x=x[:-forecast_len]

y=np.array(no2['NO2'])
y=y[:-forecast_len]

sc = MinMaxScaler(feature_range=(0,1))
x = sc.fit_transform(x)

x_train = []
y_train = []

for i in range(3,500):
    x_train.append(x[i-3:i,0])
    y_train.append(x[i,0])
    
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_train,y_train = np.array(x_train),np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(128,return_sequences=True , input_shape = (x_train.shape[1],1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(512,return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(512,return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(512,return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(256,return_sequences=True),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.LSTM(128,return_sequences=True),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer = 'adam',
             loss = 'mean_squared_error',
             metrics=['accuracy'])

model.summary()

history = model.fit(x_train,y_train,epochs=15,batch_size=3)
predictions = model.predict(x_train)
#print(predictions)

plt.plot(range(len(x_train)), y_train, c='g')
plt.plot(range(len(x_train)), predictions, c='r')
plt.legend(['Green-Train', 'Red-Predictions'], loc='upper left')
plt.show()
