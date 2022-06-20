import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os
import cv2
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import random

directory = os.path.dirname(os.path.realpath(__file__))
directory = os.path.join(directory,'train1K')

classes = {'Healthy':0,'Black Sigatoka':1}

data = []
X = []
Y = []

for cls in classes:
    pth = os.path.join(directory,cls)
    for j in os.listdir(pth):
        img = cv2.imread(os.path.join(pth,j),0)
        img = cv2.resize(img, (256,256))
        data.append([img,classes[cls]])
    X.append(img)
    Y.append(classes[cls])

random.shuffle(data)
for feature, label in data:
    X.append(feature)
    Y.append(label)

X = np.array(X)
Y = np.array(Y)

X_updated = X.reshape(len(X), -1)
np.unique(Y)
pd.Series(Y).value_counts()
X.shape, X_updated.shape
X_updated = X.reshape(len(X), -1)
X_updated.shape
xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10,test_size=0.25)
xtrain.shape, xtest.shape
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())

print(xtrain.shape, xtest.shape)

pca = PCA(.98)
# pca_train = pca.fit_transform(xtrain)
# pca_test = pca.transform(xtest)
pca_train = xtrain
pca_test = xtest




#lg = LogisticRegression(C=0.1)
#lg.fit(xtrain, ytrain)
sv = SVC()
sv.fit(xtrain, ytrain)

pickle.dump(sv,open('model.p','wb'))
print("Pickle is dumped successfully")


#print("Training Score:", lg.score(xtrain, ytrain))
#print("Testing Score:", lg.score(xtest, ytest))

print("Training Score:", sv.score(xtrain, ytrain))
print("Testing Score:", sv.score(xtest, ytest))


