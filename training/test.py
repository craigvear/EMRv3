
# import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np


model = load_model('models/EMR-3_RNN_skeleton_data.nose.x.h5')

for x in range(100):
    x /= 100
    input = np.reshape(x, (1, 1, 1))
    pred = model.predict(input)
    print(pred)

