# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio
from scipy.sparse import coo_matrix
import pickle


def _load_network(filename, mtrx='adj'):
    print("### Loading [%s]..." % (filename))
    if mtrx == 'adj':
        i, j, val = np.loadtxt(filename).T
        #A = coo_matrix((val, (i-1,j-1)),shape=(239,239))
        #A = coo_matrix((val, (i-1,j-1)),shape=(67,67))
        A = coo_matrix((val, (i-1,j-1)),shape=(882,882))#the number of featuer drug: 882, protein: 1449
        #A = coo_matrix((val, (i-1,j-1)),shape=(1449,1449))
        A = A.todense()
        A = np.squeeze(np.asarray(A))
        if A.min() < 0:
            print("### Negative entries in the matrix are not allowed!")
            A[A < 0] = 0
            print("### Matrix converted to nonnegative matrix.")
            print
        if (A.T == A).all():
            pass
        else:
            print("### Matrix not symmetric!")
            A = A + A.T
            print("### Matrix converted to symmetric.")
    else:
        print("### Wrong mtrx type. Possible: {'adj', 'inc'}")
    A = A - np.diag(np.diag(A))
    A = A + np.diag(A.sum(axis=1) == 0)

    return A


def load_networks(filenames, mtrx='adj'):
    Nets = []
    for filename in filenames:
        Nets.append(_load_network(filename, mtrx))

    return Nets


def _net_normalize(X):
    """
    Normalizing networks according to node degrees.
    """
    if X.min() < 0:
        print("### Negative entries in the matrix are not allowed!")
        X[X < 0] = 0
        print("### Matrix converted to nonnegative matrix.")
        print
    if (X.T == X).all():
        pass
    else:
        print("### Matrix not symmetric.")
        X = X + X.T - np.diag(np.diag(X))
        print("### Matrix converted to symmetric.")

    # normalizing the matrix
    deg = X.sum(axis=1).flatten()
    deg = np.divide(1., np.sqrt(deg))
    deg[np.isinf(deg)] = 0
    D = np.diag(deg)
    X = D.dot(X.dot(D))

    return X


def net_normalize(Net):
    """
    Normalize Nets or list of Nets.
    """
    if isinstance(Net, list):
        for i in range(len(Net)):
            Net[i] = _net_normalize(Net[i])
    else:
        Net = _net_normalize(Net)

    return Net


def _scaleSimMat(A):
    """Scale rows of similarity matrix"""
    A = A - np.diag(np.diag(A))
    A = A + np.diag(A.sum(axis=0) == 0)
    col = A.sum(axis=0)
    A = A.astype(np.float)/col[:, None]

    return A


def RWR(A, K=3, alpha=0.9):
    """Random Walk on graph"""
    A = _scaleSimMat(A)
    # Random surfing
    n = A.shape[0]
    P0 = np.eye(n, dtype=float)
    P = P0.copy()
    M = np.zeros((n, n), dtype=float)
    for i in range(0, K):
        P = alpha*np.dot(P, A) + (1. - alpha)*P0
        M = M + P

    return M


def PPMI_matrix(M):
    """ Compute Positive Pointwise Mutual Information Matrix"""
    M = _scaleSimMat(M)
    n = M.shape[0]
    col = np.asarray(M.sum(axis=0), dtype=float)
    col = col.reshape((1, n))
    row = np.asarray(M.sum(axis=1), dtype=float)
    row = row.reshape((n, 1))
    D = np.sum(col)

    np.seterr(all='ignore')
    PPMI = np.log(np.divide(D*M, np.dot(row, col)))
    PPMI[np.isnan(PPMI)] = 0
    PPMI[PPMI < 0] = 0

    return PPMI


if __name__ == "__main__":
    path_to_string_nets = '../test_data/drug/'#the path of drug nets
    #path_to_string_nets = '../test_data/protein/'#the path of protein nets
    string_nets = ['drug_strc','drug_se','drug_drug','drug_disease']#all nets of drug
    #string_nets = ['protein_seq','protein_protein','protein_disease']#all nets of protein
    filenames = []
    for net in string_nets:
        filenames.append(path_to_string_nets + 'Sim_mat_' + net +'.txt')

    # Load STRING networks
    Nets = load_networks(filenames)
    # Compute RWR + PPMI
    for i in range(0, len(Nets)):
        #Nets[i]=net_normalize(Nets[i])
        print("### Computing PPMI for network: %s" % (string_nets[i]))
        Nets[i] = RWR(Nets[i])
        Nets[i] = PPMI_matrix(Nets[i])
        #Nets[i]=net_normalize(Nets[i])
        print("### Writing output to file...")
        sio.savemat('../test_data/drug/Sim_mat_K3_alpha0.9_' + string_nets[i] +'.mat',{'Net':Nets[i]})#save drug nets
        #sio.savemat('../test_data/protein/Sim_mat_K3_alpha0.9_' + string_nets[i] +'.mat',{'Net':Nets[i]})#save protein nets
        print('save '+string_nets[i]+' end')
