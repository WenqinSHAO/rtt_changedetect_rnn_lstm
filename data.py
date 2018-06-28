import os
import re
import pandas as pd
import numpy as np

def cpt_seg_color(x):
    """use cpt to color a time series, e.g.
    x = [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
    y = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]

    Args:
        x(one dimension interable)

    Returs:
        y(numpy array)
    """
    y = [0]
    for prev, now in zip(x[:-1],x[1:]):
        if now == 1:
            y.append(1-y[-1])
        else:
            y.append(y[-1])
    return np.array(y)

def color_to_cpt(x):
    """convert seg coloring (above) back to changepoints
    """
    y=[0]
    for prev, now in zip(x[:-1], x[1:]):
        if now != prev:
            y.append(1)
        else:
            y.append(0)
    return np.array(y)

def load_data(path, sep=',', decimal='.', color=True, x_key='trace', y_key='cpt'):
    """load data from a given folder

    Args:
        path (str): folder containing data
        sep (str): seperation symbol of the csv file
        decimal (str): deciaml symbol of the csv file
        color (bool): whether turn changepoints into colored segments in main_y (returns)
        x_key (str): column name for input
        y_key (str): column name for output

    Return:
        x (numpy.array): shape of (#_sequence, #_unique_seq_lenght), or (#_sequence,) in case of different length
        aux_y (numpy.array): 1 if x eperienced at least one change, 0 the otherwise
        main_y (numpy.array): for each datapoint in a sequence, 1 for a changepoint, 0 otherwise
    """
    fn_pattern = re.compile("^[0-9]+.csv$")
    if not os.path.isdir(path):
        print("%s is not a directory"%path)
        return

    x = []
    aux_y = []
    main_y = []
    files = []

    for file in os.listdir(path):
        if fn_pattern.match(file):
            files.append(file)
    files = sorted(files, key=lambda s: int(s.split('.')[0]))

    for file in files:
        d = pd.read_csv(os.path.join(path,file), sep=sep, decimal=decimal)
        x.append(np.array(d[x_key]))
        # 1 if the input expeirenced at least one change
        aux_y.append([1 if np.sum(np.array(d[y_key])) > 0 else 0])
        # color the sequence according to the changepoints or not
        if color:
            main_y.append(cpt_seg_color(np.array(d[y_key])))
        else:
            main_y.append(np.array(d[y_key]))

    x = np.array(x)
    aux_y = np.array(aux_y)
    main_y = np.array(main_y)
    if len(x.shape) > 1:
        x = x.reshape(x.shape[0], x.shape[1])
        aux_y = aux_y.reshape(aux_y.shape[0], aux_y.shape[1])
        main_y = main_y.reshape(main_y.shape[0], main_y.shape[1])
    # number of sequences experienced at least one change
    print(sum(aux_y))
    return x, aux_y, main_y
    
def submin(x):
    """substract the min value for each given timesereis
    """
    xx = [[i - max(0, min(xi)) for i in xi] for xi in x]
    xx = np.array(xx)
    xx = xx.reshape(xx.shape[0], xx.shape[1])
    return xx

def delta(x):
    """cal the difference between two neighbouring datapoint
    """
    xx = []
    for xi in x:
        xxx = [0]
        for prev, now in zip(xi[:-1], xi[1:]):
            xxx.append(abs(now-prev))
        xx.append(xxx)
    xx = np.array(xx)
    xx = xx.reshape(xx.shape[0], xx.shape[1])
    return xx