from matplotlib import pyplot
from pylab import *
from numpy.linalg import det
import numpy as np
import numpy.matlib as ml
import random

def DataGeneration(nSamples):
    mean = [20,15]
    cov = [[30,0],[0,110]]
    data = np.random.multivariate_normal(mean,cov,nSamples).T
    return data

def InitParams(centers,k):
    pMiu = centers
    pPi = zeros([1,k], dtype=float)
    pSigma = zeros([len(Data[0]), len(Data[0]), k], dtype=float)
    dist = Distmat(Data, centers)
    labels = dist.argmin(axis=1)
    for j in range(k):
        idx_j = (labels == j).nonzero()
        pMiu[j] = Data[idx_j].mean(axis=0)
        pPi[0, j] = 1.0 * len(Data[idx_j]) / Number_of_Samples
        pSigma[:, :, j] = cov(mat(Data[idx_j]).T)
    return pMiu, pPi, pSigma

def Distmat(X, Y):
    n = len(X)
    m = len(Y)
    xx = ml.sum(X*X, axis=1)
    yy = ml.sum(Y*Y, axis=1)
    xy = ml.dot(X, Y.T)
    return tile(xx, (m, 1)).T+tile(yy, (n, 1)) - 2*xy



def CalProb(k,pMiu,pSigma):
    Px = zeros([Number_of_Samples, k], dtype=float)
    for i in range(k):
        Datashift = mat(Data - pMiu[i, :])
        inv_pSigma = mat(pSigma[:, :, i]).I
        coef = sqrt(2*3.14*det(mat(pSigma[:, :, i])))            
        for j in range(Number_of_Samples):
            tmp = (Datashift[j, :] * inv_pSigma * Datashift[j, :].T)
            Px[j, i] = 1.0 / coef * exp(-0.5*tmp)
    return Px


    
def eStep(Px, pPi):
    pGamma =mat(array(Px) * array(pPi))
    pGamma = pGamma / pGamma.sum(axis=1)
    return pGamma

def mStep(pGamma):
    Nk = pGamma.sum(axis=0)
    pMiu = diagflat(1/Nk) * pGamma.T * mat(Data) 
    pSigma = zeros([len(Data[0]), len(Data[0]), Number_of_Gau], dtype=float)
    for j in range(Number_of_Gau):
        Datashift = mat(Data) - pMiu[j, :]
        for i in range(Number_of_Samples):
            pSigmaK = Datashift[i, :].T * Datashift[i, :]
            pSigmaK = pSigmaK * pGamma[i, j] / Nk[0, j]
            pSigma[:, :, j] = pSigma[:, :, j] + pSigmaK
    return pGamma, pMiu, pSigma

def PylabPlot(X, labels, iter, k):
    colors = ["orange","green","pink","coral","yellow","blue"]
    pyplot.plot(hold=False)
    
    labels = array(labels).ravel()
    data_colors=[colors[lbl] for lbl in labels]
    pyplot.scatter(X[:, 0], X[:, 1], c=data_colors, alpha=None, linewidths=0)
    pyplot.savefig('iter_%02d.png' % iter, format='png')

def MoG(X, k, threshold=1e-10):
    N = len(X)
    labels = zeros(N, dtype=int)
    centers = array(random.sample(X, k))
    iter = 0
    pMiu, pPi, pSigma = InitParams(centers,k)
    Lprev = float('-10000')
    pre_esp = 100000
    while iter < 100:
        Px = CalProb(k,pMiu,pSigma)
        pGamma = eStep(Px, pPi)
        pGamma, pMiu, pSigma = mStep(pGamma)
        labels = pGamma.argmax(axis=1)
        L = sum(log(mat(Px) * mat(pPi).T))
        cur_esp = L-Lprev
        if cur_esp < threshold:
            break
        if cur_esp > pre_esp:
            break
        pre_esp=cur_esp
        Lprev = L
        iter += 1
    PylabPlot(X, labels, iter, k)

# Parameters
Number_of_Samples = 100
Number_of_Gau = 3

# MoG
sample_data = DataGeneration(Number_of_Samples)
Data=array(mat(sample_data).T)
MoG(Data, Number_of_Gau)

