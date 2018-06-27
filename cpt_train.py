import numpy as np
import benchmark as bch
import data
import model as md

if __name__ == '__main__':
    
    x, aux_y, main_y = data.load_data("train_data/")

    # basic data preprocessing
    x = data.submin(x)
    x_seq = x.reshape(x.shape[0], x.shape[1], 1)
    
    N_SAMPLE = x.shape[0]
    N_BATCH = 5000
    N_EPOCH = 200

    model = md.model_1()
    history = model.fit({'input_seq':x_seq, 'input_array':x}, 
                        {'aux_out':aux_y, 'main_out':main_y},
                        validation_split=0.2,
                        epochs=N_EPOCH, batch_size=N_BATCH, verbose=1)

    #model = md.model_2()
    #history = model.fit(x_seq, aux_y, validation_split=0.2, 
    #                    epochs=N_EPOCH, batch_size=N_BATCH, verbose=1)

    md.save_trained_model(model, fn='cpt_model_1')
    md.plot_leanring_curv(history, fn='cpt_model_1')