from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing
from matplotlib import cm
import matplotlib.pyplot as plt
from dataLoader import *

#if __name__ == '__main__':
def train():
    trainfolder = ["trainA512","trainA768"]
    maxsize = [256, 512]
    directory = os.path.dirname(os.path.realpath(__file__))
    for z in trainfolder:
        for s in maxsize:
            string = str(s) + "-" + str(z)
            folder = os.path.join(directory, "ARCHIVE")
            folder = os.path.join(folder, string)
            data_train, cat_train = load_data(z,s)
            CreateDirectory(folder)
            np.save(os.path.join(folder, "data_train.npy"), data_train)
            np.save(os.path.join(folder, "cat_train.npy"), cat_train)
            print("Saved Vector")
