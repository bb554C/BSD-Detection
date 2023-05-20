from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.feature import local_binary_pattern
from sklearn import preprocessing
from matplotlib import cm
import matplotlib.pyplot as plt
from dataLoader import *

if __name__ == '__main__':
    af = "testOrig"
    maxsize =  512
    Category = ["Black Sigatoka", "Healthy"]
    neighbor = 2
    datafile = "data_train.npy"
    catfile = "cat_train.npy"
    
    data_train = np.load(datafile)
    cat_train = np.load(catfile)

    le = preprocessing.LabelEncoder()
    le.fit(cat_train)
    label_train = le.transform(cat_train)
    vec_test, cat_test = load_data(af, maxsize)
    label_test = le.transform(cat_test)

    #KNN
    model = KNeighborsClassifier(n_neighbors=neighbor, n_jobs=-1)
    model.fit(data_train, label_train)

    #EVALUATE
    prediction = model.predict(vec_test)
    count = 1
    cat = 0
    for pred in prediction:
        print(count,". Actual ",Category[cat],"Predicted ",pred," - ", Category[pred])
        if(count < 30):
            count = count + 1
        else:
            cat = cat + 1
            count = 1
        
    cmat = get_conf_mat(y_pred=prediction, y_target=label_test,n_cats=len(le.classes_))
    acc = cmat.trace() / cmat.shape[0]
    print("Accuracy ", acc)
