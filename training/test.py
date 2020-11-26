import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np


model = load_model('models/EMR-3_RNN_bitalino.h5')

for x in range(100):
    # x /= 100
    input = np.reshape(x, (x.shape[1], x.shape[0], 1))

    x = model.predict(0.1)
    print(x)
