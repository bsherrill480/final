__author__ = 'brian'
from LoadData import GetData
import cv2
from scipy.cluster.vq import *
import numpy as np
import pylab

class MyKMeans:
    def __init__(self, imageList, sampStart = 1, sampEnd = 62):
        doNum = 10
        data = GetData(sampleStart=sampStart, sampleEnd=sampEnd)
        letters = data.GetLetters()
        self.imgs = imageList
        #avgs10 = data.GetAveraged()[0:10]
        avgs = data.GetAveraged()
        avgs = avgs[sampStart-1, sampEnd]
        self.Kguesses = np.array([list(j.flatten()) for j in avgs])
        self.k = sampEnd - sampStart + 1
    def ObsAndKInt(self):
        obs = np.array([list(j.flatten()) for i in self.imgs for j in i])
        return kmeans2(obs, self.k)
    def WhitenObsAndKInt(self):
        obs = np.array([list(j.flatten()) for i in self.imgs for j in i])
        obs = whiten(obs)
        return kmeans2(obs, self.k)
    def ObsAndKGuess(self):
        obs = np.array([list(j.flatten()) for i in self.imgs for j in i]) #flattens each image into a row
        return kmeans2(obs, self.kguesses)
    def WhitenObsAndKWhitenGuess(self):
        obsAndK = np.array([list(j.flatten()) for i in self.imgs for j in i] + [list(j.flatten()) for j in self.avgs])
        obsKWhit = whiten(obsAndK)
        #print len(obsKWhit)
        n = len(obsKWhit)
        o = len(self.kguesses)
        return kmeans2(obsKWhit[0:n-o], obsKWhit[n-o:n])
        print res
        print "------------"
        print idx
    def CheckCorrects (self, res, idx):
        resList = list(idx)
        for i in resList:
            pass
    def GetKGuesses(self, imageList):
        pass


if __name__ == "__main__":
    #build lists of letter (training data)

    """
    doNum = 10
    data = GetData(sampleStart=1, sampleEnd=10)
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
    n = len(obsKWhit)
    cents, idx = kmeans2(obsKWhit[0:n-10], obsKWhit[n-10:n])
    print cents.shape
    print "------------"
    print str(list(idx))
    ###
    #try whiten
    #no whiten + kguesses
    #or whiten + no k = 10
    #obsW = whiten(obs)#consider playing around with
    """
    """
    res, idx = kmeans2(obsW, Kguesses)
    print res
    print "------------"
    print idx
    """