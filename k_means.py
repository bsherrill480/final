__author__ = 'brian'
from LoadData import GetData
import cv2
from scipy.cluster.vq import *
import numpy as np
import pylab

class MyKMeans:
    def __init__(self, sampStart = 1, sampEnd = 62):
        doNum = 10
        data = GetData(sampleStart=sampStart, sampleEnd=sampEnd)
        letters = data.GetLetters()
        self.imgs = data.GetListImgs()
        #avgs10 = data.GetAveraged()[0:10]
        avgs = data.GetAveraged()
        self.avgs = avgs[sampStart-1: sampEnd]
        self.kguesses = np.array([list(j.flatten()) for j in avgs])
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
    def RoughErrors (self, idx):
        """
        Expects 55 is each catagory. Counts the number of items from the 55
        """
        counts = np.zeros(self.k, dtype=np.int_)
        idxList = list(idx)
        #print idxList
        for i in idxList:
            counts[i] = counts[i] + 1
        return sum([abs(55 - i) for i in counts]) / 2

    def ComplexErrors(self, idx):
        """
        Finds most common cluster id in set of same letters, and then uses that as the true ID of the set of letters
        and counts the number that do not belong
        """
        counts = np.zeros(self.k, dtype=np.int_)
        itter = 0
        idxList = list(idx)
        mostCommonInBlock = []
        for i in idxList:
            counts[i] = counts[i]+1
            if itter%54 == 0:
                itter = -1
                indexOfLargest = 0
                maxVal = counts[0]
                for j in xrange(len(counts)):
                    if counts[j] > maxVal:
                        maxVal = counts[j]
                        indexOfLargest = j
                mostCommonInBlock.append(counts[j])
                counts[:] = 0
            itter = itter+1
        errors = 0
        itter = 0
        mostCommonIndex = 0
        for i in idxList:
            if counts[i] != mostCommonInBlock[mostCommonIndex]:
                errors = errors + 1
            if itter % 54 == 0:
                itter = -1
                mostCommonIndex = mostCommonIndex + 1
            itter = itter + 1
        return errors


if __name__ == "__main__":
    #Play around to see what technique gives best result
    Kmeans = MyKMeans(4,10)
    CE,CL1 = Kmeans.WhitenObsAndKInt()
    CE,CL2 = Kmeans.ObsAndKInt()
    CE,CL3 = Kmeans.ObsAndKGuess()
    CE,CL4 = Kmeans.WhitenObsAndKInt()

    L = [CL1, CL2, CL3, CL4]
    for i in L:
        print "Complex Errors: " + str(Kmeans.ComplexErrors(i))

    for i in L:
        print "Rough Errors: " + str(Kmeans.RoughErrors(i))
