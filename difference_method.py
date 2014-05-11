import numpy as np
import os
import cv2
class difference_method():
    def BuildListLetters(self):
        dirWithPics = "/home/brian/Pictures/LETTERS/"
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        imagesList = [cv2.imread(dirWithPics + letters[i] + ".jpg", 0) for i in xrange(26)]
        '''
        for i in imagesList:
            cv2.imshow('image',i)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        '''
        return imagesList
    def NormalizeImage(self, image):
        colSum = np.sum(image, axis = 0) #columns
        rowSum = np.sum(image, axis = 1) #rows
        colStart, colEnd = self.begginingEndIndex(colSum)

    def BegginingEndIndex(self, array):
        begin = 0
        end = array.size
        inLetter = array[0] == 0 #test to see if we're startin in letter
        for i in xrange(array.size):
            if array[i] > begin:
                begin = array[i]

if __name__ == "__main__":
    dm = difference_method()
    imagesList = dm.BuildListLetters()
    print np.sum(imagesList[0], axis = 0).size