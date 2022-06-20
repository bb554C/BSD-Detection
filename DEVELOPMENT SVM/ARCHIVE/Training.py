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
from sklearn.svm import SVC
import pickle
import datetime

e = datetime.datetime.now()
Categories=['Healthy','Black Sigatoka']

directory = os.path.dirname(os.path.realpath(__file__))
directory = os.path.join(directory,'train1K')

flat_data_arr=[]
target_arr=[]
for i in Categories:
  print(f'loading... category : {i}')
  path=os.path.join(directory,i)
  for img in os.listdir(path):
    img_array=imread(os.path.join(path,img))
    img_resized=resize(img_array,(256,256,3))
    flat_data_arr.append(img_resized.flatten())
    target_arr.append(Categories.index(i))
    #print(target_arr)
  print(f'loaded category: {i} successfully')
print(len(flat_data_arr))
flat_data=np.array(flat_data_arr)
target=np.array(target_arr)
df=pd.DataFrame(flat_data)
df['Target']=target
df

x=df.iloc[:,:-1]
y=df.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state = 77,stratify=y)
print("Dimensions of input training data:",x_train.shape)
print("Dimensions of input testing data:",x_test.shape)
print("Dimensions of output training data:",y_train.shape)
print("Dimensions of output testing data:",y_test.shape)
print('Splitted Successfully')

print ("The time is: = %s:%s:%s" % (e.hour, e.minute, e.second))
param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.01],'kernel':['rbf']}
svc=svm.SVC(probability=True)
print("The training of the model is started, please wait for while as it may take few minutes to complete")
model=GridSearchCV(svc,param_grid)
model.fit(x_train,y_train)
print('The Model is trained well with the given images')
print(model.best_params_)

pickle.dump(model,open('img_model.p','wb'))
print("Pickle is dumped successfully")
"""
y_pred=model.predict(x_test)
print("The predicted Data is :")
y_pred

print("The actual data is:")
np.array(y_test)

#classification_report(y_pred,y_test)
print(f"The model is, {accuracy_score(y_pred,y_test)*100}% accurate")
#confusion_matrix(y_pred,y_test)`1

"""
