
# import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

class Net:
    def __init__(self):
        print('net init')

    def model(self):
        self.model = load_model('models/EMR-3_RNN_skeleton_data.nose.x.h5')

    def predict(self, inval):
        pred = self.model.predict(inval)
        return pred

net = Net()
net.model()
for x in range(100):
    x /= 100
    inval = np.reshape(x, (1, 1, 1))
    pred = net.predict(inval)
    print(pred)

