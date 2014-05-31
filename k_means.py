__author__ = 'brian'
from LoadData import GetData
import cv2
from scipy import cluster
import numpy as np
if __name__ == "__main__":
    #build lists of letter (training data)
    data = GetData(doLess=10)
    letters = data.GetLetters()
    imgs = data.GetListImgs()
    obs = np.array([list(j.flatten()) for i in imgs for j in i]) #flattens each image into a row
    

