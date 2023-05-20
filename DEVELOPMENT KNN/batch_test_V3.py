from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.feature import local_binary_pattern
from sklearn import preprocessing
from matplotlib import cm
import matplotlib.pyplot as plt
from dataLoader import *

#if __name__ == '__main__':
def test():
    trainfolder = ["trainA512","trainA768"]
    af = ["testOrig"]
    maxsize = [256, 512]
    accu = 0
    neigh = 0
    path = ""
    Category = ["Healthy","Black Sigatoka"]
    neighbor = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    directory = os.path.dirname(os.path.realpath(__file__))
    for x in maxsize:
        for t in trainfolder:
            accu = 0
            dataStringFolder = str(x) + "-" + t
            dataFolder = os.path.join(directory, "ARCHIVE")
            dataFolder = os.path.join(dataFolder, dataStringFolder)
            datafile = os.path.join(dataFolder, "data_train.npy")
            catfile = os.path.join(dataFolder, "cat_train.npy")
            data_train = np.load(datafile)
            cat_train = np.load(catfile)
            for s in maxsize:
                print("TestSize: ",s)
                le = preprocessing.LabelEncoder()
                le.fit(cat_train)
                label_train = le.transform(cat_train)
                vec_test, cat_test = load_data(af[0], s)
                label_test = le.transform(cat_test)

                #KNN
                for n in neighbor:
                    model = KNeighborsClassifier(n_neighbors=n, n_jobs=-1)
                    model.fit(data_train, label_train)
                    prediction = model.predict(vec_test)
                    cmat = get_conf_mat(y_pred=prediction, y_target=label_test,n_cats=len(le.classes_))
                    acc = cmat.trace() / cmat.shape[0]
                    print("Neigbor ", n, " Accuracy", acc)
                    if(acc > accu):
                        accu = acc
                        neigh = n
                        msize = s
            print("Accuracy: ", accu, " Neighbors: ",neigh,"Image Test Size:",msize)
            fstring = "{:.4f}".format(accu) + " n" + str(neigh).zfill(2) + " " + str(msize).zfill(4) + " " + dataStringFolder
            ffolder = os.path.join(directory, "ARCHIVE")
            ffolder = os.path.join(ffolder, fstring)
            CreateDirectory(ffolder)
            MoveFiles(dataFolder, ffolder)
            RemoveDirectories(dataFolder)
