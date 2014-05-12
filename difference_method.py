import numpy as np
import os
import cv2
class build_normalized_list:

    def __init__(self, pathsToLetters):
        """
        input: takes list of strings (paths) to letters s.t. A is index 0, B is index 1...
            Images must be contain letters with white background and black letters
        output: normalized 16x16 letters where black is the background and white are letters
        """
        self.imagesList = [self.NormalizeImage( cv2.imread(i, 0) ) for i in pathsToLetters]

    def GetList(self):
        return self.imagesList

    def DisplayImageList(self, imagesList):
        """
        debuggin to examine images
        """
        for i in imagesList:
            print i
            cv2.imshow('image',i)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def NormalizeImage(self, image):
        """
        turns whites to blacks and Normalizes pictures to 8x8 or 16x16 (im still debating)
        """
        image = self.InvertImage(image)
        colSum = np.sum(image, axis = 0) #columns
        rowSum = np.sum(image, axis = 1) #rows
        sumTot = np.sum(colSum)
        tolerance = .005 #percent tolerance
        colStart, colEnd = self.BegginingEndIndex(colSum, tolerance)
        rowStart, rowEnd = self.BegginingEndIndex(rowSum, tolerance)
        return cv2.resize(image[rowStart : rowEnd+1, colStart : colEnd+1], (16, 16)) #+1 to end because slicing is [inclusive, exclusive]

    def InvertImage(self, image):
        allWhite = np.ndarray(shape = image.shape, dtype=np.uint8)
        allWhite[:] = np.uint8(255)
        return cv2.subtract(allWhite, image)

    def BegginingEndIndex(self, array, tolerance):
        """
        tolerance is contained in (0,1) % of totalTot
        """
        toleranceNum = np.sum(array)*tolerance
        begin = 0
        end = array.size - 1 #zero based indexing
        inLetter = not (array[0] < toleranceNum) #test to see if we're starting in letter
        for i in xrange(array.size):
            if array[i] > toleranceNum and (not inLetter):
                begin = i
                inLetter = True
            if array[i] < toleranceNum and inLetter:
                end = i
                break
        #print "array size: " + str(array.size)
        #print "array: " + str(array)
        #print "being, end: " + str((begin, end))
        return (begin, end)

class PatternMatcher:
    def __init__(self, list, ):


if __name__ == "__main__":
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    listOfDirsWithLetters = ["/home/brian/Pictures/LETTERS/"] #different directories
    listOfNormalizedLetters = []
    for dirWithPics in listOfDirsWithLetters:
        pathToLetters = [dirWithPics + letters[i] + ".jpg" for i in xrange(26)]#generates list of paths to each letter
        listOfNormalizedLetters.append(build_normalized_list(pathToLetters).GetList())
    #display letters
    for j in listOfNormalizedLetters:
        for i in j:
            cv2.imshow('image',i)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
