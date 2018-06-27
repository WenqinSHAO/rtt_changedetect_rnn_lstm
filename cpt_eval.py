import numpy as np
import data
import model as md
import benchmark as bch
import changedetect as chpt
import changedetectRNN as chpt_rnn
import matplotlib.pyplot as plt

if __name__ == "__main__":

   
    # load artificial validation data set
    x, aux_y, main_y = data.load_data("valid_data/")

    # pre-process x as when they get trained
    # basic data preprocessing
    x = data.submin(x)
    x_seq = x.reshape(x.shape[0], x.shape[1], 1)

    # model 1 against artifical validation date set
    model = md.load_model("cpt_model_1")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    # all loss, aux_y loss, main_y loss, aux_y acc, main_y acc
    print(model.evaluate({'input_seq':x_seq, 'input_array':x}, 
                        {'aux_out':aux_y, 'main_out':main_y}))

    # model 2 against artificial validation data set
    model = md.load_model("cpt_model_2")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    # loss, acc
    print(model.evaluate(x_seq, aux_y))

    # model 1 against real RTT timeseries, compared to changepoint.np
    X, _, Y = data.load_data("real_trace_labelled/", 
                           sep=';', decimal='.', color=False,
                           x_key='rtt', y_key='cp')

    precision_rnn = []
    precision_np = []
    recall_rnn = []
    recall_np = []

    for x, y in zip(X, Y):
        truth = [i for i, v in enumerate(y) if v == 1]
        
        res = chpt_rnn.cpt_rnn(x)
        pred = [i for i, v in enumerate(res) if v == 1]
        eva = bch.evaluation_window_adp(truth, pred, window=3)
        precision_rnn.append(eva['precision'])
        recall_rnn.append(eva['recall'])

        pred = chpt.cpt_np(x)
        eva = bch.evaluation_window_adp(truth, pred, window=3)
        precision_np.append(eva['precision'])
        recall_np.append(eva['recall'])


    plt.scatter(precision_rnn, recall_rnn, c='g', marker='v', alpha=0.6, label='LSTM')
    plt.scatter(precision_np, recall_np, c='r', marker='o', alpha=0.6, label='Bayesian NP')
    plt.xlabel('precision')
    plt.ylabel('recall')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()

