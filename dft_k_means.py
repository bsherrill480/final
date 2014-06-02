__author__ = 'brian'
from LoadData import GetData
import cv2
from scipy.cluster.vq import *
import numpy as np
import os

class MyDFTKMeans:
    def __init__(self, sampStart = 11, sampEnd = 36):
        data = GetData(sampleStart=sampStart, sampleEnd=sampEnd)
        self.sampStart = sampStart
        self.sampEnd = sampEnd
        self.imgs = data.GetListImgs()
        self.letters = data.GetLetters()
        avgs = data.GetAveraged()
        self.avgs = avgs[sampStart-1: sampEnd]
        self.kguesses = np.array([list(j.flatten()) for j in avgs])
        self.k = sampEnd - sampStart + 1
        self.data = data
    def ObsAndKInt(self):
        obs = np.array([list(j.flatten()) for i in self.imgs for j in i])
        return kmeans2(obs, self.k)
    def WhitenObsAndKInt(self):
        obs = np.array([list(j.flatten()) for i in self.imgs for j in i])
        obs = whiten(obs)
        return kmeans2(obs, self.k)
    def ObsAndKGuess(self):
        obs = np.array([list(j.flatten()) for i in self.imgs for j in i]) #flattens each image into a row
        #for i in range(5):
        #    print np.array(obs[i]).reshape(16,16)

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
        Expects 55 is each catagory. Counts the number of items deviating from 55 in each catagory
        """
        counts = np.zeros(self.k, dtype=np.int_)
        idxList = list(idx)
        #print idxList
        for i in idxList:
            counts[i] = counts[i] + 1
        return sum([abs(55 - i) for i in counts]) / 2

    def ComplexErrors(self, idx):
        """
        find most common member in a sample (55 of same images) and selects that as true cluster ID
        Counts number of not true cluster ID in sample
        pre: raw cluster returned from kmeans2()
        post: returns error count
        """
        errors = 0
        Samps = self.BreakIntoSamples(idx)
        for i in Samps:
            MostCommon = self.GetMostCommonElement(i)
            for j in i:
                if j != MostCommon:
                    errors = errors + 1
        return errors

    def GetMostCommonElement(self, L):
        """
        pre: gives list of elements from k possible clusters
        post:return most common element
        """

        counts = np.zeros(self.k, dtype=np.int_)
        for i in L:
            counts[i] = counts[i] + 1
        countsList = list(counts)
        #Not efficient, but simple
        return countsList.index(max(countsList))

    def BreakIntoSamples(self, idxList):
        """
        pre: samples are 55 in each: cluters passed is raw from kmeans2()
        takes in clusters, breaks into list of list of samples
        post: returns list of each element assigned to cluster
        """

        listOfSamples = []
        if not isinstance(idxList, list):
            idxList = list(idxList)
        for i in xrange(0,len(idxList), 55):
            listOfSamples.append(idxList[i:i+55])
        return listOfSamples

    def ClusterIDToLetter(self, samps):
        """
        samps is from BreakIntoSamples
        """
        dictTupes = []
        for i in xrange(self.k):
            mostCommon = self.GetMostCommonElement(samps[i])
            dictTupes.append( (mostCommon, self.letters[i+self.sampStart]) )
        print dict(dictTupes)
        return dict(dictTupes)

    def DFTKMeans(self):
        for i in self.imgs:
            for j in i:
                #get most powerful freq in image
                imageFlat = j.flatten().astype(np.float_)
                n = len(imageFlat)
                power = sum(imageFlat * imageFlat) / n
                imgHat = np.fft.fft(imageFlat) / n
                posHarmonicsPow = imgHat[0:n/2] / power
                sortedPow = posHarmonicsPow.sort()
                runningSum = 0
                values = []
                for k in xrange(len(sortedPow)):
                    values.append(sortedPow[k])
                    runningSum = runningSum + sortedPow[k]
                    if runningSum > .9:
                        break






    def ClassifyTestsM1(self, testDir = "/home/brian/PycharmProjects/firstAttempt/HandLetters"):
        """
        uses ObsAndKGuess to classify images
        returns list of strings of what file was matched to what letter
        """
        #testImages = [data.NormalizeImage(cv2.imread(testDir + "/" + j, 0)) for j in os.listdir(testDir)]
        #testImagesFlat = np.array([list(j.flatten()) for j in testImages])
        centroids, clusters = self.ObsAndKGuess()
        #Get cluster ID
        samples = self.BreakIntoSamples(clusters)
        letterDict = self.ClusterIDToLetter(samples)

        #use test data to classify
        testImages = [(self.data.NormalizeImage(cv2.imread(testDir + "/" + j, 0)),j) for j in os.listdir(testDir)]#tuple (image, letter)
        """#display images
        def DisplayImages(thing):
            if isinstance(thing, list):
                for i in thing:
                    cv2.imshow('image',i[0])
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                print"YOU DUN GOOFED"
        DisplayImages(testImages)
        """
        matches = []
        for i in testImages:
            imageArray = np.array(list(i[0].flatten())).astype(np.float_)
            name = i[1]

            #compare to centroids
            minDist= np.linalg.norm(imageArray - centroids[0].astype(np.float_))
            bestCent = 0
            for j in xrange(self.k):
                dist = np.linalg.norm(imageArray - centroids[j].astype(np.float_))
                if dist < minDist:
                    minDist = dist
                    bestCent = j
            matches.append(name + " was identified to be: " + letterDict[bestCent])
        return matches

def testMethods():
    """
    used to determine best k-means method
    """

    KmeansT = MyKMeans(sampStart=11,sampEnd=36)
    CE,CL1 = KmeansT.WhitenObsAndKInt()
    CE,CL2 = KmeansT.ObsAndKInt()
    CE,CL3 = KmeansT.ObsAndKGuess()
    CE,CL4 = KmeansT.WhitenObsAndKWhitenGuess()
    L = [CL1, CL2, CL3, CL4]

    for i in L:
        print i
    for i in L:
        print "Complex Errors: " + str(KmeansT.ComplexErrors(i))

    for i in L:
        print "Rough Errors: " + str(KmeansT.RoughErrors(i))

if __name__ == "__main__":
    #Play around to see what technique gives best result
    #testMethods()

    Kmeans = MyKMeans(11,36)

    def DisplayImages(thing):
        if isinstance(thing, list):
            for i in thing:
                cv2.imshow('image',i)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            cv2.imshow('image',thing)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    errors = 0
    items = 0
    for i in Kmeans.ClassifyTestsM1():
        print i
        if i[0] != i[-1]:
            errors = errors+1
        items = items + 1

    print str(errors) + "/" + str(items) + " errors testing (k-means)"


    #DisplayImages(testImages)
