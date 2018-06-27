import numpy as np
import time
from keras.layers import Input, Concatenate, Dot, Flatten
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.models import Model, model_from_json
from keras.utils import plot_model
import matplotlib.pyplot as plt

def save_trained_model(model,fn="model"):
    """save to file the trained model
    """
    # serialize model to JSON
    model_json = model.to_json()
    with open("%s.json"%fn, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("%s.h5"%fn)

def plot_leanring_curv(rec, fn='model'):
    """plot learning curve during training
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for k in rec.history.keys():
        ax.plot(rec.history[k])
    ax.set_xlabel('epoch')
    ax.legend(rec.history.keys(), loc='upper left')
    fig.set_size_inches(10,8)
    plt.savefig("%s_learning_curve.pdf"%fn, format='pdf')
    plt.close()

def load_model(fn):
    """load model from file
    """
    model_json = open("%s.json"%fn, 'r')
    model = model_json.read()
    model_json.close()
    model = model_from_json(model)
    model.load_weights("%s.h5"%fn)
    return model

def model_1():
    """model for the detection of change points
    """
    # inputs; works only for sequences of 100 datapoint
    input_seq = Input(shape=(100,1), name='input_seq')
    input_array = Input(shape=(100,), name='input_array')

    lstm = LSTM(100, return_sequences=True)(input_seq)

    # auxiliary out only tells where there is change or not
    # the main out tell where the change happens

    deep = TimeDistributed(Dense(20, activation='relu'))(lstm)
    deep = Flatten()(deep)
    aux_out = Dense(1, activation='sigmoid', name='aux_out')(deep)

    deep = Concatenate()([deep, input_array])
    deep = Dense(50, activation='relu')(deep)
    deep = Dense(50, activation='relu')(deep) 
    main_out = Dense(100, activation='sigmoid', name='main_out')(deep)
    
    model = Model(inputs=[input_seq, input_array], outputs=[aux_out, main_out])
    model.compile(loss={'aux_out':'binary_crossentropy', 'main_out':'binary_crossentropy'},
                  loss_weights = {'aux_out': 0.1, 'main_out': 1.0},
                  optimizer='adam')

    print(model.summary())
    plot_model(model, show_shapes=True, to_file='cpt_model_1.png')

    return model

def model_2():
    """mode for the detection whether there is change in given timesereis
    """
    inputs = Input(shape=(None,1))  # should work for seq of arbitrary length
    lstm = LSTM(100)(inputs)
    out = Dense(1, activation='sigmoid')(lstm)
    model = Model(inputs=inputs, outputs=out)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    print(model.summary())
    plot_model(model, show_shapes=True, to_file='cpt_model_2.png')
    return model




