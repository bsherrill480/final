import numpy as np
import os
import cv2
class build_normalized_list:

    def __init__(self, pathsToLetters, debug = False):
        """
        input: takes list of strings (paths) to letters s.t. A is index 0, B is index 1...
            Images must be contain letters with white background and black letters
        output: normalized 16x16 letters where black is the background and white are letters
        """
        self.debugs = debug
        if debug:
            print "debug on"
        self.imagesList = [self.NormalizeImage( cv2.imread(i, 0) ) for i in pathsToLetters]

    def GetList(self):
        return self.imagesList


    def NormalizeImage(self, image):
        """
        turns whites to blacks and Normalizes pictures to 8x8 or 16x16 (im still debating)
        """
        image = self.InvertImage(image)
        colSum = np.sum(image, axis = 0) #columns
        rowSum = np.sum(image, axis = 1) #rows
        sumTot = np.sum(colSum)
        tolerance = .005 #percent tolerance
        colStart, colEnd = self.BegginingEndIndex2(colSum, tolerance)
        rowStart, rowEnd = self.BegginingEndIndex2(rowSum, tolerance)
        return cv2.resize(image[rowStart : rowEnd+1, colStart : colEnd+1], (16, 16)) #+1 to end because slicing is [inclusive, exclusive]

    def InvertImage(self, image):
        allWhite = np.ndarray(shape = image.shape, dtype=np.uint8)
        allWhite[:] = np.uint8(255)
        return cv2.subtract(allWhite, image)

    def BegginingEndIndex2(self, array, tolerance):
        """
        tolerance is contained in (0,1) % of totalTot
        """
        toleranceNum = np.sum(array)*tolerance
        if self.debugs:
            print "tol num "+str(toleranceNum)
            print array
        begin = 0
        end = array.size - 1 #zero based indexing
        for i in xrange(array.size):
            if array[i] > toleranceNum:
                begin = i
                break
        i = end
        while(i >= 0 ):
            if array[i] > tolerance:
                end = i
                break
            i = i-1
        if self.debugs:
            print (begin, end)

        return (begin, end)

class display_images:
    def display(self, thing):
        if isinstance(thing, list):
            for i in thing:
                cv2.imshow('image',i)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            cv2.imshow('image',thing)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

class pattern_matcher:
    def __init__(self, listOfImages, img):
        self.image = img
        self.listOfImages = listOfImages
        diff = self.differenceArray()
        min = diff[0]
        self.index = 0
        print diff
        for i in xrange(len(listOfImages)):
            if diff[i] < min:
                min = diff[i]
                self.index = i
    def differenceArray(self):
        return [np.sum(cv2.subtract(i, self.image)) for i in self.listOfImages]

    def getMatch(self):
        return self.index
if __name__ == "__main__":
    #build lists of letter
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    listOfDirsWithLetters = ["/home/brian/Pictures/LETTERS/"] #different directories
    listOfNormalizedLetters = []
    for dirWithPics in listOfDirsWithLetters:
        pathToLetters = [dirWithPics + letters[i] + ".jpg" for i in xrange(26)]#generates list of paths to each letter
        listOfNormalizedLetters.append(build_normalized_list(pathToLetters).GetList())
    #normalize hand drawn A
    Apath = "/home/brian/Pictures/A_test.jpg"
    ANorm = build_normalized_list([Apath], True).GetList()[0]
    d = display_images()
    print ANorm
    d.display(ANorm)
    #d.display(listOfNormalizedLetters[0])
    #A = build_normalized_list(listOfNormalizedLetters)#wtf is this?
    indexOfMatch = pattern_matcher(listOfNormalizedLetters[0], ANorm).getMatch()
    print "Anorm was matched to: " + str(letters[indexOfMatch])
    '''
    #display letters
    for j in listOfNormalizedLetters:
        for i in j:
            cv2.imshow('image',i)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    '''
