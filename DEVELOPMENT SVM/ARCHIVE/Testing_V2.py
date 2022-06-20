import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

categories=['Healthy','Black Sigatoka','Unknown']

directory = os.path.dirname(os.path.realpath(__file__))

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

random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.20)

pick = open('model.save','rb')
model = pickle.load(pick)

prediction = model.predict(xtest)

accuracy = model.score(xtest, ytest)

print('Accuracy: ', accuracy)
print('Prediction is: ',categories[prediction[0]])

leaves = xtest[0].reshape(256,256)
plt.imshow(leaves, cmap='gray')
plt.show()
