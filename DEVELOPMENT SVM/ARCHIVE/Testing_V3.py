import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os
import cv2

modelName = 'model_V3_2.p'
categoryNumber = 1

sv = pickle.load(open(modelName,'rb'))

Categories = {0:'Healthy',1:'Black Sigatoka'}

directory = os.path.dirname(os.path.realpath(__file__))
testsource = os.path.join(directory,"test")
testsource = os.path.join(testsource,Categories[categoryNumber])

counter = 0
for filename in os.listdir(testsource):
    print(filename)
    imgpath = os.path.join(testsource,filename)
    img = cv2.imread(imgpath,0)
    img1 = cv2.resize(img, (256,256))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    print(Categories[p[0]])
    if(Categories[categoryNumber] == Categories[p[0]]):
        counter = counter + 1
print("Correct Items: ",counter)
