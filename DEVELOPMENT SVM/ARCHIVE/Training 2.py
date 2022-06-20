import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

directory = os.path.dirname(os.path.realpath(__file__))
directory = os.path.join(directory,'train')

categories=['Healthy','Black Sigatoka','Unknown']

data = []

for category in categories:
    path = os.path.join(directory,category)
    label = categories.index(category)
    
    for img in os.listdir(path):
        imgpath = os.path.join(path,img)
        leaf_img = cv2.imread(imgpath,0)
        try:
            leaf_img = cv2.resize(leaf_img,(256,256))
            image = np.array(leaf_img).flatten()
            data.append([image,label])
        except Exception as e:
            pass

print(len(data))

pick_in = open('data1.pickle','wb')
pickle.dump(data,pick_in)
pick_in.close()

random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.98)

model = SVC(C=1, kernel='poly', gamma='auto',probability=True)
model.fit(xtrain,ytrain)

pick = open('model.save','wb')
pickle.dump(model,pick)
pick.close()
