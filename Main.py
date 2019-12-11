import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from random import shuffle

DATADIR = "C:/MLSets/PetImages"
categories = ["Dog", "Cat"]
X = []
y = []
TrainingData = []
imgSize = 50

try:
    X = np.load('features.npy')
    y = np.load('label.npy')

except Exception:
    pass


def createtrainingdata():
    # Loops through categories ["Dog", "Cat"]
    for category in categories:
        # Joining the data and the path and getting the class value
        path = os.path.join(DATADIR, category)
        classNum = categories.index(category)
        for img in os.listdir(path):
            # Making all images the same and prepping for insertion
            try:
                img_array = cv2.imread(os.path.join(path, img),
                                       cv2.IMREAD_GRAYSCALE)  # Path&Img Joinig, Making Grayscale

                newArray = cv2.resize(img_array, (imgSize, imgSize))  # Resizing to be 50x50
                TrainingData.append([newArray, classNum]) # Appending to the fixed array
            except Exception:
                pass


# If the program hasn't been run before
if X == [] and y == []:
    createtrainingdata()
    shuffle(TrainingData)  # Shuffling to make random

    for features, label in TrainingData:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, imgSize)

np.save('features.npy', X)
np.save('label', y)
