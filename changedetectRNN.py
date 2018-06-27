import data
import model as md

fullDetect = md.load_model('cpt_model_1')
binaryDetect = md.load_model('cpt_model_2')

def cpt_rnn(x, detector='full'):
    # only supports full Detect now
    model = fullDetect
    # cut the input into chunks of 100 datapoints
    x_cut = cut(x)
    # pre-processing
    x_cut = data.submin(x_cut)
    x_seq = x_cut.reshape(x_cut.shape[0], x_cut.shape[1], 1)

    # perform detection for each single chunck
    res = model.predict({'input_seq':x_seq, 'input_array':x_cut})
    cpt = []
    for binary, seg_col in zip(res[0], res[1]):
        # when the model is confident about the occurence of changes, add up the changepoints
        if binary[0] > 0.5:
            seg_col = [1 if v > 0.5 else 0 for v in seg_col]
            cpt += list(data.color_to_cpt(seg_col))
        else:
            cpt += ([0]*len(seg_col))
    
    return cpt[:len(x)]
    

def cut(x, l=100):
    """cut a a given sequence into pieces of same length
    """
    x = list(x)
    if len(x) < l:
        x = padd(x, l)
        return [x]
    
    remain = len(x)
    xx = []
    i = 0
    while remain >= l:
        xx.append(x[i:i+l])
        i += l
        remain -= l
    if remain:
        xx.append(padd(x[i:],l))
    return xx


def padd(x, l):
    """padd x to length l with its last value
    """
    
    assert len(x) < l
    x += [x[-1]] * (l-len(x))
    return x