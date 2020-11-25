
import tensorflow as tf
import numpy as np
import pandas as pd
from random import randrange
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from scipy import stats

# load the dataset from csv into panda dataframe
df = pd.read_csv(r'test_craig_vear_20201124.csv', header=0)

# name the columns for easy filtering
col_name = ['_id', 'session_id', 'session_name', 'timestamp', 'delta_time',
       'audio_file', 'mic_volume', 'video_file', 'backing_track_file',
       'bitalino', 'brainbit.eeg-T3', 'brainbit.eeg-T4', 'brainbit.eeg-O1',
       'brainbit.eeg-O2', 'skeleton_data.nose.x', 'skeleton_data.nose.y',
       'skeleton_data.nose.confidence', 'skeleton_data.neck.x',
       'skeleton_data.neck.y', 'skeleton_data.neck.confidence',
       'skeleton_data.r_shoudler.x', 'skeleton_data.r_shoudler.y',
       'skeleton_data.r_shoudler.confidence', 'skeleton_data.r_elbow.x',
       'skeleton_data.r_elbow.y', 'skeleton_data.r_elbow.confidence',
       'skeleton_data.r_wrist.x', 'skeleton_data.r_wrist.y',
       'skeleton_data.r_wrist.confidence', 'skeleton_data.l_shoudler.x',
       'skeleton_data.l_shoudler.y', 'skeleton_data.l_shoudler.confidence',
       'skeleton_data.l_elbow.x', 'skeleton_data.l_elbow.y',
       'skeleton_data.l_elbow.confidence', 'skeleton_data.l_wrist.x',
       'skeleton_data.l_wrist.y', 'skeleton_data.l_wrist.confidence',
       'skeleton_data.r_eye.x', 'skeleton_data.r_eye.y',
       'skeleton_data.r_eye.confidence', 'skeleton_data.l_eye.x',
       'skeleton_data.l_eye.y', 'skeleton_data.l_eye.confidence',
       'skeleton_data.r_ear.x', 'skeleton_data.r_ear.y',
       'skeleton_data.r_ear.confidence', 'skeleton_data.l_ear.x',
       'skeleton_data.l_ear.y', 'skeleton_data.l_ear.confidence', 'flow_level',
       'date', 'last_update']
df.columns = col_name

# filter the ones I need
df = df.filter(['date', 'bitalino', 'brainbit.eeg-T3', 'brainbit.eeg-T4', 'brainbit.eeg-O1',
       'brainbit.eeg-O2', 'skeleton_data.nose.x', 'skeleton_data.nose.y'])

# use only 1 recording
df = df[:1500]

# show box distribution
# sns.boxplot(x=df['bitalino'])
# sns.boxplot(x=df['brainbit.eeg-T4'])
# plt.show()

# scatter plot
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df['bitalino'], df['brainbit.eeg-T4'])
ax.set_xlabel('bitalino')
ax.set_ylabel('brainbit.eeg-T4')
plt.show()


# normalise all columns
column_names_to_normalize = ['bitalino', 'brainbit.eeg-T3', 'brainbit.eeg-T4', 'brainbit.eeg-O1',
       'brainbit.eeg-O2', 'skeleton_data.nose.x', 'skeleton_data.nose.y']
x = df[column_names_to_normalize].values
min_max_scaler = preprocessing.MaxAbsScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = df.index)
df[column_names_to_normalize] = df_temp

# df["Time"] = pd.to_datetime(df['date'])
# df.plot(x="Time", y=['bitalino', 'brainbit.eeg-T4'])
# plt.show()


# remove outliers
# df_o = df.filter()
z = np.abs(stats.zscore(df['bitalino', 'brainbit.eeg-T3', 'brainbit.eeg-T4', 'brainbit.eeg-O1',
       'brainbit.eeg-O2', 'skeleton_data.nose.x', 'skeleton_data.nose.y']))
threshold = 3
print(np.where(z > 3))

df = df[(z < 3).all(axis=1)]

df["Time"] = pd.to_datetime(df['date'])
df.plot(x="Time", y=['bitalino', 'brainbit.eeg-T4'])
plt.show()

#


#
# # calc correlations across the fields
# corr = df.corr()
# print(corr)
#
# # Plotting the pairplot of correlation between features
# sns.pairplot(df, height=1.5)
# plt.show()
#
# # plot the heatmap
# sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
# plt.show()

# # apply filter function
# # A) just the features we want
# # df = df.filter(['x', 'y', 'z', 'freq', 'amp'])
# # B) all rows with amp values greater than 0.1 (i.e. has a sound) & freq below 2500
# # df = df[df['amp'] > 0.1] # ignoring this for now as I want non-events to be learnt
# df = df[df['freq'] < 2500]
# #  OPTIONAL C) only "body" data
# df = df[df['limb'] == '/Body']
# df = df.filter(['x', 'y', 'z', 'amp']) # just the operational data
#
# dataset = df.values # just the values
# print (dataset)
#
# # 3. split into train and test sets
# train_size = int(len(dataset) * 0.67) # 67% Train
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
#
#
# # convert an array of values into a dataset matrix
# def create_dataset(dataset, n_future=1, n_past=1):
#     x_train = []
#     y_train = []
#
#     for i in range (n_past, len(dataset)-n_past-1):
#         # for num in range(n_past):
#         x_train.append(dataset[i][:-1]) # incoming array
#         y_train.append(dataset[i][:3]) # linked to output prediction
#         # print(x_train, y_train)
#     x_train, y_train = np.array(x_train), np.array(y_train)
#     print(x_train.shape[0])
#     x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#     return np.array(x_train), np.array(y_train)
#
#
#     # dataX, dataY = [], []
#     # for i in range(len(dataset)-look_back-1):
#     #     a = dataset[i:(i+look_back), 0]
#     #     print('a   ', a)
#     #     dataX.append(a)
#     #     print('X   ', dataX)
#     #     dataY.append(dataset[i + look_back, 0])
#     #     print('Y   ', dataY)
#     # return numpy.array(dataX), numpy.array(dataY)
#
# # 4. reshape into X=t and Y=t+5
# n_future = 1  # next 4 events todo sort out lookback
# n_past = 10  # Past 30 events
# trainX, trainY = create_dataset(train, n_future, n_past)
# testX, testY = create_dataset(test, n_future, n_past)
#
# # reshape input to be [samples, time steps, features]
# # trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1)) # 83480 features, 5 time steps, 5 features
# # print (trainX.shape[0], trainX.shape[1], trainX.shape[2])
# # testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
#
# # 5. create and fit the LSTM network
# model = tf.keras.models.Sequential()
# # metrics = tf.keras.metrics.Accuracy(name='accuracy', dtype=None)
#
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True), input_shape=(trainX.shape[1], 1))) # input shape is 5 timesteps, 1-3d feature
# model.add(tf.keras.layers.Dropout(0.2))
# model.add(tf.keras.layers.LSTM(64, return_sequences=True))
# model.add(tf.keras.layers.Dropout(0.2))
# model.add(tf.keras.layers.LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension
# model.add(tf.keras.layers.Dropout(0.2))
# model.add(tf.keras.layers.LSTM(64))
# model.add(tf.keras.layers.Dropout(0.2))
# model.add(tf.keras.layers.Dense(units=3)) # how many outputs as predictions
#
# model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=tf.keras.metrics.Accuracy(name='accuracy', dtype=None))
# model.fit(trainX, trainY, epochs=200, batch_size=32, verbose=1)
#
# model.save('LSTM_Bidirectional_64x4_no_lookback_200epochs-AMPin-XYout_model.h5')
# print ('saved')
#
