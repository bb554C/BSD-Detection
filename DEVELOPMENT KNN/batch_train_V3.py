from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing
from matplotlib import cm
import matplotlib.pyplot as plt
from dataLoader import *

#def train():
if __name__ == '__main__':
    #
    trainfolder = ["trainA512","trainA768"]
    #
    maxsize = [256, 512]
    directory = os.path.dirname(os.path.realpath(__file__))
    for z in trainfolder:
        for s in maxsize:
            string = str(s) + "-" + str(z)
            folder = os.path.join(directory, "ARCHIVE")
            folder = os.path.join(folder, string)
            data_train, cat_train = load_data(z,s)
            #data_train = data_train.reshape(data_train.shape[0], maxsize*maxsize*3)
            CreateDirectory(folder)
            np.save(os.path.join(folder, "data_train.npy"), data_train)
            np.save(os.path.join(folder, "cat_train.npy"), cat_train)
            print("Saved Vector")
