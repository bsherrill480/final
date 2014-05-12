import numpy as np
import os
import cv2
class difference_method():
    def BuildListLetters(self):
        """
        builds list of alphabet pictures from /home/brian/Pictures/LETTERS/
        """
        dirWithPics = "/home/brian/Pictures/LETTERS/"
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        imagesList = [cv2.imread(dirWithPics + letters[i] + ".jpg", 0) for i in xrange(26)]
        return imagesList

    def DisplayImageList(self, imagesList):
        """
        debuggin to examine images
        """
        for i in imagesList:
            cv2.imshow('image',i)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def NormalizeImage(self, image):
        """
        turns whites to blacks and Normalizes pictures to 8x8 or 16x16 (im still debating)
        """
        colSum = np.sum(image, axis = 0) #columns
        rowSum = np.sum(image, axis = 1) #rows
        colStart, colEnd = self.begginingEndIndex(colSum)
        rowStart, rowEnd = self.begginingEndIndex(rowSum)
        image = image[]

    def InvertImage(self, image):
        allWhite = np.ndarray(shape = image.shape, dtype=np.uint8)
        allWhite[:] = np.uint8(255)
        return cv2.subtract(allWhite, image)

    def BegginingEndIndex(self, array):
        """
        NO TOLLERANCE SET. change for tollerance if necesary
        """
        begin = 0
        end = array.size - 1 #zero based indexing
        inLetter = not (array[0] == 0) #test to see if we're starting in letter
        for i in xrange(array.size):
            if array[i] > 0 and (not inLetter):
                begin = i
                inLetter = True
            if array[i] == 0 and inLetter:
                end = i
                break
        return (begin, end)


if __name__ == "__main__":
    dm = difference_method()
    imagesList = dm.BuildListLetters()
    dm.DisplayImageList(imagesList)