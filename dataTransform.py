import os
from random import shuffle

import cv2
import numpy as np

DATADIR = "C:/MLSets/PetImages"
categories = ["Dog", "Cat"]
x1 = []
y1 = []
x2 = []
y2 = []
x4 = []
y4 = []
x6 = []
y6 = []
x8 = []
y8 = []
x12 = []
y12 = []

TrainingData = []
imgSize = 50


def createtrainingdata():
    # Loops through categories ["Dog", "Cat"]
    for category in categories:
        lp = 0
        # Joining the data and the path and getting the class value
        path = os.path.join(DATADIR, category)
        classNum = categories.index(category)
        for img in os.listdir(path):
            lp += 1
            # Making all images t   he same and prepping for insertion
            try:
                img_array = cv2.imread(os.path.join(path, img),
                                       cv2.IMREAD_GRAYSCALE)  # Path&Img Joinig, Making Grayscale

                newArray = cv2.resize(img_array, (imgSize, imgSize))  # Resizing to be 50x50
                TrainingData.append([newArray, classNum])  # Appending to the fixed array
            except Exception:
                pass

            if lp == 1000:
                for features, label in TrainingData:
                    x1.append(features)
                    y1.append(label)

            if lp == 2000:
                for features, label in TrainingData:
                    x2.append(features)
                    y2.append(label)

            if lp == 4000:
                for features, label in TrainingData:
                    x4.append(features)
                    y4.append(label)

            if lp == 6000:
                for features, label in TrainingData:
                    x6.append(features)
                    y6.append(label)

            if lp == 8000:
                for features, label in TrainingData:
                    x8.append(features)
                    y8.append(label)

            if lp == 12000:
                for features, label in TrainingData:
                    x12.append(features)
                    y12.append(label)


allvars = [x1, x2, x4, x6, x8, x12, y1, y2, y4, y6, y8, y12]
createtrainingdata()
for i in allvars:
    shuffle(i)

np.save('features1.npy', x1)
np.save('labels1.npy', y1)

np.save('features2.npy', x2)
np.save('labels2.npy', y2)

np.save('features4.npy', x4)
np.save('labels4.npy', y4)

np.save('features6.npy', x6)
np.save('labels6.npy', y6)

np.save('features8.npy', x8)
np.save('labels8.npy', y8)

np.save('features12.npy', x12)
np.save('labels12.npy', y12)
