__author__ = 'brian'
from LoadData import GetData
import cv2
from scipy.cluster.vq import *
import numpy as np
import os

class ValueIndexWrapper:
    def __init__(self, value, index):
        self.value = value
        self.index = index
    def __lt__(self, other):
        return self.value.__lt__(other.value)
    def __le__(self, other):
        return self.value.__le__(other.value)
    def __eq__(self,other):
        return self.value.__eq__(other.value)
    def __ne__(self,other):
        return self.value.__ne__(other.value)
    def __gt__(self,other):
        return self.value.__gt__(other.value)
    def __ge__(self,other):
        return self.value.__ge__(other.value)
    def __repr__(self):
        return "(" + str(self.value) +" @index: " + str(self.index) + ")"

class MyDFTKMeansBleed:
    def __init__(self, sampStart = 11, sampEnd = 36):
        data = GetData(sampleStart=sampStart, sampleEnd=sampEnd)
        self.sampStart = sampStart
        self.sampEnd = sampEnd
        self.imgs = data.GetListImgs()
        self.letters = data.GetLetters()
        self.avgs = data.GetOneOfEach() #THIS IS NO LONGER AVERAGES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #self.avgs = avgs[sampStart-1: sampEnd]
        len(self.avgs)
        #self.kguesses = np.array([list(j.flatten()) for j in avgs])#changed for DFT
        self.k = sampEnd - sampStart + 1
        self.data = data

        ##Change for DFT
        self.kguesses = self.GetDFTAvgs()
        self.obs = self.GetDFTObs()
    def ObsAndKInt(self):
        obs = self.obs
        return kmeans2(obs, self.k)
    def WhitenObsAndKInt(self):
        obs = self.obs
        obs = whiten(obs)
        return kmeans2(obs, self.k)
    def ObsAndKGuess(self):
        obsB = self.Bleed(self.obs) #flattens each image into a row
        avgsB = self.Bleed(self.GetDFTAvgs())
        return kmeans2(obsB, avgsB)

    def Bleed(self, M):
        wM = .9#each index has equal weight. wM= weightMe
        wN = .05 #weightNeighbor
        returnMe = np.zeros(M.shape)
        for i in xrange(len(M)):
            for j in xrange(len(M[i])):
                if j == 0:
                    returnMe[i,j] = wM*M[i,j] +wN*M[i,j+1]
                elif j == len(M[i]) - 1:
                    returnMe[i,j] = wM*M[i,j] +wN*M[i,j-1]
                else:
                    returnMe[i,j] = wM*M[i,j] +wN*M[i,j-1] +wN*M[i,j+1]
        return returnMe
    def WhitenObsAndKWhitenGuess(self):
        obsList = []
        for i in self.imgs:
            for j in i:
                #get most powerful freq in image
                imageFlat = j.flatten().astype(np.float_)
                n = len(imageFlat)
                powerTot = sum(imageFlat * imageFlat) / n #parsival's theorem
                imgDFT = np.fft.fft(imageFlat) / n
                posHarmonicsPow = (np.abs(imgDFT[1:n/2+1])**2) / powerTot # change to 0 TO ADD MEAN BACK!!!!!!!!!!!!!!!!!!!!!!!!!
                #wrap it with index then sort it
                wrappedPow = [ValueIndexWrapper(posHarmonicsPow[i], i) for i in xrange(len(posHarmonicsPow))]
                sortedPow = sorted(wrappedPow)
                runningSum = 0
                #reduce down to just the powerful harmonics
                powerfulHarmonics = np.zeros(n/2, dtype=np.float_) #n/2+1 since we include mean (A[0])#CHANGE TO n/2 +1 TO ADD MEAN BACK !!!!!!!!!!!
                for k in xrange(len(sortedPow)):
                    powerfulHarmonics[sortedPow[-k].index] = sortedPow[-k].value
                    runningSum = runningSum + sortedPow[-k].value
                    if runningSum > .5:
                        #print str(k) + " harmonic terms used"#super spam
                        break
                obsList.append(powerfulHarmonics)

        for j in self.avgs:

            #get most powerful freq in image
            imageFlat = j.flatten().astype(np.float_)
            n = len(imageFlat)
            powerTot = sum(imageFlat * imageFlat) / n #parsival's theorem
            imgDFT = np.fft.fft(imageFlat) / n

            obsList.append((np.abs(imgDFT[1:n/2 + 1])**2) / powerTot) #change to 0 TO ADD MEAN BACK!!!!!!!!!!!!!!!!!!!!1

        obsKWhit = np.array(obsList)
        n = len(obsKWhit)
        o = len(self.GetDFTAvgs())
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

    def GetDFTObs(self):
        obsList = []
        for i in self.imgs:
            for j in i:
                #get most powerful freq in image
                imageFlat = j.flatten().astype(np.float_)
                n = len(imageFlat)
                powerTot = sum(imageFlat * imageFlat) / n #parsival's theorem
                imgDFT = np.fft.fft(imageFlat) / n
                posHarmonicsPow = (np.abs(imgDFT[1:n/2+1])**2) / powerTot # change to 0 TO ADD MEAN BACK!!!!!!!!!!!!!!!!!!!!!!!!!
                #wrap it with index then sort it
                wrappedPow = [ValueIndexWrapper(posHarmonicsPow[i], i) for i in xrange(len(posHarmonicsPow))]
                sortedPow = sorted(wrappedPow)
                runningSum = 0
                #reduce down to just the powerful harmonics
                powerfulHarmonics = np.zeros(n/2, dtype=np.float_) #n/2+1 since we include mean (A[0])#CHANGE TO n/2 +1 TO ADD MEAN BACK !!!!!!!!!!!
                for k in xrange(len(sortedPow)):
                    powerfulHarmonics[sortedPow[-k].index] = sortedPow[-k].value
                    runningSum = runningSum + sortedPow[-k].value
                    if runningSum > .5:
                        #print str(k) + " harmonic terms used"#super spam
                        break
                obsList.append(powerfulHarmonics)
        print np.array(obsList).shape
        return np.array(obsList)
    def GetDFTAvgs(self): #The programmer in me dies using copy paste, but the this is just for a demonstration
        L = []
        #print len(self.avgs)
        for j in self.avgs:

            #get most powerful freq in image
            imageFlat = j.flatten().astype(np.float_)
            n = len(imageFlat)
            powerTot = sum(imageFlat * imageFlat) / n #parsival's theorem
            imgDFT = np.fft.fft(imageFlat) / n

            L.append((np.abs(imgDFT[1:n/2 + 1])**2) / powerTot) #change to 0 TO ADD MEAN BACK!!!!!!!!!!!!!!!!!!!!1
            # posHarmonicsPow = (np.abs(imgDFT[0:n/2])**2) / powerTot
            #wrap it with index then sort it
            #just use raw DFT of avgs
            x = 1#USED TO SELECT TO KEEP OR TRIM GUESSES
            if x == 1:
                L.pop()
                posHarmonicsPow = (np.abs(imgDFT[1:n/2+1])**2) / powerTot # change to 0 TO ADD MEAN BACK!!!!!!!!!!!!!!!!!!!!!!!!!

                wrappedPow = [ValueIndexWrapper(posHarmonicsPow[i], i) for i in xrange(len(posHarmonicsPow))]
                sortedPow = sorted(wrappedPow)
                runningSum = 0
                #reduce down to just the powerful harmonics
                powerfulHarmonics = np.zeros(n/2, dtype=np.float_) #n/2+1 since we include mean (A[0])
                for k in xrange(len(sortedPow)):
                    powerfulHarmonics[sortedPow[-k].index] = sortedPow[-k].value
                    runningSum = runningSum + sortedPow[-k].value
                    if runningSum > .9:
                        #print str(k) + " harmonic terms used"#super spam
                        break
                L.append(powerfulHarmonics)

        print np.array(L).shape
        return np.array(L)






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
        print letterDict
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
            imageFlat = i[0].flatten().astype(np.float_)
            n = len(imageFlat)
            powerTot = sum(imageFlat * imageFlat) / n #parsival's theorem
            imgDFT = np.fft.fft(imageFlat) / n

            imageArray = self.Bleed( np.array([  (np.abs(imgDFT[1:n/2 + 1])**2) / powerTot ]) )#CHANGE TO 0 TO ADD MEAN BACK!!!!!!!!!!!!!!!!!!!!
            name = i[1]

            #compare to centroids
            minDist= np.linalg.norm(imageArray - centroids[0].astype(np.float_))
            bestCent = 0
            for j in xrange(self.k):
                dist = np.linalg.norm(imageArray - centroids[j].astype(np.float_))
                if dist < minDist:
                    minDist = dist
                    bestCent = j

            try:
                matches.append(name + " was identified to be: " + letterDict[bestCent])
            except:
                matches.append(name + " was identified to be centroid: " + str(bestCent))

        return matches

def testMethods():
    """
    used to determine best k-means method
    """

    DFTKmeansT = MyDFTKMeansBleed(sampStart=11,sampEnd=36)
    #CE,CL1 = DFTKmeansT.WhitenObsAndKInt()
    #CE,CL2 = DFTKmeansT.ObsAndKInt()
    CE,CL3 = DFTKmeansT.ObsAndKGuess()
    #print CE
    #print CL3
    CE,CL4 = DFTKmeansT.WhitenObsAndKWhitenGuess()
    #L = [CL1, CL2, CL3, CL4]
    L = [CL3, CL4]
    #L = [CL3]
    for i in L:
        print i
    for i in L:
        print "Complex Errors: " + str(DFTKmeansT.ComplexErrors(i))

    for i in L:
        print "Rough Errors: " + str(DFTKmeansT.RoughErrors(i))

if __name__ == "__main__":
    #Play around to see what technique gives best result
    #testMethods()
    #testMethods()

    DFTKmeans = MyDFTKMeansBleed(11,36)

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
    for i in DFTKmeans.ClassifyTestsM1():
        print i
        if i[0] != i[-1]:
            errors = errors+1
        items = items + 1

    print str(errors) + "/" + str(items) + " errors testing (k-means)"

    #DisplayImages(testImages)
