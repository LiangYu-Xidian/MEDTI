# coding=UTF-8
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale, scale
from validation import cross_validation, temporal_holdout
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping
from MEDTI import build_MDA

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os.path as Path
import scipy.io as sio
import numpy as np
import pickle
import sys
import pandas as pd


def read_params(fname):
    params = {}
    fR = open(fname, 'r')
    for line in fR:
        print (line.strip())
        key, val = line.strip().split('=')
        key = str(key.strip())
        val = str(val.strip())
        if key == 'select_arch' or key == 'select_nets':
            params[key] = map(int, val.strip('[]').split(','))
        else:
            params[key] = str(val)
    print ("###############################################################")
    print
    print
    fR.close()

    return params


def build_model(X, input_dims, arch, nf=0.5, std=1.0, mtype='mda', epochs=80, batch_size=64,mid_num=0):#arch:Number of neurons per layer
    if mtype == 'mda':
        model = build_MDA(input_dims, arch)
    else:
        print ("### Wrong model.")
    # corrupting the input
    noise_factor = nf
    if isinstance(X, list):
        Xs = train_test_split(*X, test_size=0.2)
        X_train = []
        X_test = []
        for jj in range(0, len(Xs), 2):
            X_train.append(Xs[jj])
            X_test.append(Xs[jj+1])
        X_train_noisy = list(X_train)
        X_test_noisy = list(X_test)
        for ii in range(0, len(X_train)):
            X_train_noisy[ii] = X_train_noisy[ii] + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_train[ii].shape)
            X_test_noisy[ii] = X_test_noisy[ii] + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_test[ii].shape)
            X_train_noisy[ii] = np.clip(X_train_noisy[ii], 0, 1)
            X_test_noisy[ii] = np.clip(X_test_noisy[ii], 0, 1)
    else:
        X_train, X_test = train_test_split(X, test_size=0.2)
        X_train_noisy = X_train.copy()
        X_test_noisy = X_test.copy()
        X_train_noisy = X_train_noisy + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_train.shape)
        X_test_noisy = X_test_noisy + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_test.shape)
        X_train_noisy = np.clip(X_train_noisy, 0, 1)
        X_test_noisy = np.clip(X_test_noisy, 0, 1)
    # Fitting the model
    history = model.fit(X_train_noisy, X_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                        validation_data=(X_test_noisy, X_test),
                        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5)])
    mid_model = Model(inputs=model.input, outputs=model.get_layer('dense_'+str(mid_num)).output)

    return mid_model, history


# ### Main code starts here
params = read_params('MEDTI_params.txt')


org = params['org']   # {drug or protein}
model_type = params['model_type']  # Choose a model
ofile_keywords = params['ofile_keywords']  # {example: 'final_result'}

models_path = params['models_path']  # directory with models
results_path = params['results_path']  # directotry with results
select_arch = params['select_arch']  # The parameters of the number of neural network layers and the number of neurons in each layer(see below)
epochs = int(params['epochs'])
batch_size = int(params['batch_size'])
nf = float(params['noise_factor'])  # nf > 0 for denoising MDA
K = params['K']  # Random walk step sum length
alpha = params['alpha']  # Restart probability of restarting random walk


# all possible combinations for architectures
arch = {}
arch['mda'] = {}
arch['mda']['protein'] = {}
arch['mda']['drug'] = {}

arch['mda']['protein'] = {1: [3*1200, 2500, 3*1200],
                          2: [3*1000, 400, 3*1000],
                          3: [3*1000, 100, 3*1000],
                          4: [3*1200, 800, 800, 800, 3*1200],
                          5: [3*800, 3*500, 400, 3*500, 3*800],
                          6: [3*1000, 3*500, 100, 3*500, 3*1000],
                          7: [3*1000, 3*800, 3*600, 400, 3*600, 3*800, 3*1000],
                          8: [3*1000, 3*600, 3*400, 200, 3*400, 3*600, 3*1000],
                          9: [3*1000, 3*600, 3*300, 100, 3*300, 3*600, 3*1000],
                          10: [3*1200, 3*1000, 3*800, 3*600, 400, 3*600, 3*800, 3*1000, 3*1200],
                          11: [3*1200, 3*1000, 3*800, 3*400, 200, 3*400, 3*800, 3*1000, 3*1200],
                          12: [3*1200, 3*900, 3*600, 3*300, 100, 3*300, 3*600, 3*900, 3*1200]
                        }
protein_mid_num=[4,4,4,5,5,5,6,6,6,7,7,7]#Number of layers where the middle layer features are located

arch['mda']['drug'] = {1: [4*600, 1500, 4*600],
                       2: [1*500, 100, 1*500],
                       3: [4*300, 100, 4*300],
                       4: [4*600, 500, 400, 500, 4*600],
                       5: [4*500, 4*300, 100, 4*300, 4*500],
                       6: [4*500, 4*100, 50, 4*100, 4*500],
                       7: [4*500, 4*400, 4*300, 200, 4*300, 4*400, 4*500],
                       8: [4*500, 4*300, 4*200, 100, 4*200, 4*300, 4*500],
                       9: [4*500, 4*300, 4*100, 50, 4*100, 4*300, 4*500]
                        }
drug_mid_num=[5,5,5,6,6,6,7,7,7]#Number of layers where the middle layer features are located

# load PPMI matrices
Nets = []
input_dims = []
#select_nets=['drug_strc','drug_se','drug_drug','drug_disease']#Drug Network Name Collection
select_nets=['protein_seq','protein_protein','protein_disease']#Protein Network Name Collection
for i in select_nets:
    print ("### Loading network [%s]..." % (i))
    N = sio.loadmat('test_data/' + org + '/Sim_mat_K3_alpha0.9_' + str(i) + '.mat', squeeze_me=True)
    Net=np.asmatrix(N['Net'])
    print('Net shape:',Net.shape)
    print ("Net %s, NNZ=%d \n" % (i, np.count_nonzero(Net)))
    Nets.append(minmax_scale(Net))
    input_dims.append(Net.shape[1])
    print('----------------------')
    

# Training MDA/AE
model_names = []
for a in select_arch:
    dim_len=arch[model_type][org][a][(int)(len(arch[model_type][org][a])/2)]
    print('dim len is:'+str(dim_len))
    ceng_num=(int)(len(arch[model_type][org][a])/2)+1
    print('dim ceng_num is:'+str(ceng_num))
    #mid_num=drug_mid_num[a-1]#The number of required drug features
    mid_num=protein_mid_num[a-1]#The number of required protein features
    print('mid num:',mid_num)
    print ("### [%s] Running for architecture: %s" % (model_type, str(arch[model_type][org][a])))
    model_name = org + '_' + 'MEDTI_' + model_type.upper() + '_arch_' + str(a) + '_' + ofile_keywords + '.h5'
    if not Path.isfile(models_path + model_name):
        mid_model, history = build_model(Nets, input_dims, arch[model_type][org][a], nf, 1.0, model_type, epochs, batch_size,mid_num)
        print('model.out type:',type(mid_model))
        # svae middle layer feature
        features=mid_model.predict(Nets)
        feat_name=org+'_arch_'+str(ceng_num)+'_'+str(dim_len)+'_K3_alpha0.9_features.txt'
        feat_path=results_path+feat_name
        f=open(feat_path,'w')
        for i in range(len(features)):
            s=str(features[i][0])
            for j in range(1,len(features[i])):
                s=s+'\t'+str(features[i][j])
            f.write(s+'\n')
        print('feature saved')

        # Export figure: loss vs epochs (history)
        plt.figure()
        plt.plot(history.history['loss'], '.-')
        plt.plot(history.history['val_loss'], '.-')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(models_path + model_name + '_loss.png', bbox_inches='tight')
    model_names.append(model_name)


