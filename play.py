"""
use LSTM to learn an artifical simple sequence pattern that depends on the previous input
"""
import numpy as np
np.random.seed(42)
import benchmark as bch
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.models import Sequential
import matplotlib.pyplot as plt
import csv
import model as md

def data_gen(x):
    """given a one dimension array of integers, output an array of same shape following certain rule
    
    Args:
        x (array of int)
    
    Returns:
        y (array of int)
    """
    y = [0, 0]
    for t2, t1, t0 in zip(x[0:-2], x[1:-1], x[2:]):
        if (t2 >= t1 and t1 >= t0) or (t2 <= t1 and t1 <= t0):
            y.append(1)
        else:
            y.append(0)
    return y


if __name__ == '__main__':
    # global config
    SEQ_LEN = 20
    N_SAMPLE = 1000
    N_EPOCH = 1000
    MTX = ['acc']

    # generate random data
    x = np.random.randint(1, 100, SEQ_LEN*N_SAMPLE).reshape(N_SAMPLE, SEQ_LEN)
    y = [data_gen(xi) for xi in x]
    y = np.array(y).reshape(N_SAMPLE, SEQ_LEN, 1)
    x = x.reshape(N_SAMPLE, SEQ_LEN, 1)

    # prepare the model
    model = Sequential()
    model.add(LSTM(10, input_shape=(None,1), return_sequences=True))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=MTX)
    history = model.fit(x,y, validation_split=0.2, epochs=N_EPOCH, batch_size=N_SAMPLE, verbose=1)
    print(model.summary())

    md.save_trained_model(model, fn='play-64')
    md.plot_leanring_curv(history, fn='play-64')
