import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import pickle
import shutil
from sklearn.model_selection import train_test_split

#print(os.path.abspath(os.getcwd()))
modelName = 'img_model.p'
categoryNumber = 0

model = pickle.load(open(modelName,'rb'))
Categories=['Healthy','Black Sigatoka']
data = []

directory = os.path.dirname(os.path.realpath(__file__))
testsource = os.path.join(directory,"test")
testsource = os.path.join(testsource,Categories[categoryNumber])
print(testsource)
counter = 0
for filename in os.listdir(testsource):
    print(filename)
    imgpath = os.path.join(testsource,filename)
    leaf_img = imread(imgpath)
    #plt.imshow(leaf_img)
    #plt.show()
    leaf_img = resize(leaf_img,(256,256,3))
    image = [leaf_img.flatten()]
    #print(model.predict(image)[0])
    probability = model.predict_proba(image)
    for ind,val in enumerate(Categories):
      print(f'{val} = {probability[0][ind]*100}%')
    print("The predicted image is : " + Categories[model.predict(image)[0]])
    if(Categories[categoryNumber] == Categories[model.predict(image)[0]]):
        counter = counter + 1
print("Correct Items: ",counter)
