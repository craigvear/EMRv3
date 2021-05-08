# import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout

class Training():
    def __init__(self):
        self.n_future = 4  # next 4 bits of data
        self.n_past = 30  # Past 30 bits


    def prep_sets(self, raw_set):
        # covert raw set into a pd dataframe
        my_df = pd.DataFrame(raw_set)

        # add column name
        my_df.columns = ['feature']

        # drop the rows that are NaN or < 0
        my_df = my_df.dropna()
        my_df = my_df[my_df['feature'] > 0]

        # reset index and make training set from only the feature I want
        my_df = my_df.reset_index(drop=True)
        dataset = my_df.values

        # split into train and test sets
        train_size = int(len(dataset) * 0.67) # 67% Train
        test_size = len(dataset) - train_size
        training_set, testing_set = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

        # Feature Scaling
        sc = preprocessing.MinMaxScaler(feature_range=(0, 1))
        training_set_scaled = sc.fit_transform(training_set)
        # print (training_set_scaled)

        return training_set_scaled

    def train(self, raw_set, label):
        print(f'########  training {label}')

        training_set_scaled = self.prep_sets(raw_set)

        # create training sets ready for ML
        x_train = []
        y_train = []

        for i in range(0, len(training_set_scaled)-self.n_past-self.n_future+1):
            x_train.append(training_set_scaled[i : i + self.n_past , 0])
            y_train.append(training_set_scaled[i + self.n_past: i + self.n_past + self.n_future, 0])
        x_train , y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # create model and train
        regressor = Sequential()

        regressor.add(Bidirectional(LSTM(units=self.n_past, return_sequences=True, input_shape = (x_train.shape[1],1) ) ))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units=self.n_past, return_sequences=True))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units=self.n_past, return_sequences=True))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units=self.n_past))
        regressor.add(Dropout(0.2))
        regressor.add(Dense(units = self.n_future,activation='linear'))
        regressor.compile(optimizer='adam', loss='mean_squared_error',metrics=['acc'])
        regressor.fit(x_train, y_train, epochs=200, batch_size=32, verbose=1 )


        # # reshape testing set
        # testing = sc.transform(testing_set)
        # testing = np.array(testing)
        # testing = np.reshape(testing, (testing.shape[1], testing.shape[0], 1))
        #
        # # Now that we have our test data ready, we can test our RNN model.
        # predict = regressor.predict(testing)
        # predict = sc.inverse_transform(predict)
        # predict = np.reshape(predict, (predict.shape[1], predict.shape[0]))
        #
        # print(f'testing set = {testing_set}')
        # print(f'prediction = {predict}')

        # save model
        regressor.save(f'/models/EMR-v4_RNN_{label}.h5')
        print (f'saved_RNN {label}')


# if __name__ == '__main__':
#        rnn = Training()
#        rnn.prep_sets('skeleton_data.nose.x', 14)
#        rnn.prep_sets('bitalino', 9)
