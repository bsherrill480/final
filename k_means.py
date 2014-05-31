__author__ = 'brian'
from LoadData import GetData
import cv2
from scipy.cluster.vq import *
import numpy as np
import pylab

if __name__ == "__main__":
    #build lists of letter (training data)
    data = GetData(doLess=10)
    letters = data.GetLetters()
    imgs = data.GetListImgs()
    avgs10 = data.GetAveraged()[0:10]
    Kguesses = np.array([list(j.flatten()) for j in avgs10])
    obs = np.array([list(j.flatten()) for i in imgs for j in i]) #flattens each image into a row
    ## MAKE DIFFERENT METHODS FOR DIFFERENT STATAGIES. IE K=10 & OBS, OBS & KGUESSES, WHITEN(OBS) & K =10, ,
    ## WHITEN(OBSK) &KGUESSES = WHITENEDKOBS[N-10,N]
    ### obs with k guesses to be whitened too
    obsAndK = np.array([list(j.flatten()) for i in imgs for j in i] + [list(j.flatten()) for j in avgs10])
    obsKWhit = whiten(obsAndK)
    print len(obsKWhit)
    n = len(obsKWhit)
    res, idx = kmeans2(obsKWhit[0:n-10], obsKWhit[n-10:n])
    print res
    print "------------"
    print idx
    ###
    #try whiten
    #no whiten + kguesses
    #or whiten + no k = 10
    #obsW = whiten(obs)#consider playing around with
    """
    res, idx = kmeans2(obsW, Kguesses)
    print res
    print "------------"
    print idx
    """