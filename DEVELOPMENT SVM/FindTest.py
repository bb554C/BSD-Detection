import Augmentor
import os
import time
import random
import shutil
from PIL import Image
from PIL import ImageFile
import torchvision.transforms as transforms
import threading
import time
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image, ImageOps
import math

from pathlib import Path

from PIL import Image, ImageOps                         # for image I/O
import numpy as np                                      # N-D array module
import matplotlib.pyplot as plt                         # visualization module
# color map for confusion matrix
from matplotlib import cm

# open source implementation of LBP
from skimage.feature import local_binary_pattern
# data preprocessing module in scikit-learn
from sklearn import preprocessing
# SVM implementation in scikit-learn
from sklearn.svm import LinearSVC
import pickle
plt.rcParams['font.size'] = 11

# LBP function params
radius = 3
n_points = 8 * radius
METHOD = 'uniform'
n_bins = n_points + 2
testfolder = 'test'
    
def CreateDirectory(dir):
    try:
        os.mkdir(dir)
        #print("Folder Created: ",dir)
    except:
        pass
        #print("Folder Already Exist.",dir)

def RenamingImages(source, destination, classification, maxsize,pad):
    #print("RENAMING START")
    files = os.listdir(destination)
    count = 0
    for filename in os.listdir(source):
        if filename.endswith(".PNG") or filename.endswith(".png") or filename.endswith(".JPG") or filename.endswith(".jpg"):
            tmp_dir = os.path.join(source,filename)
            img = Image.open(tmp_dir)
            box = img.getbbox()
            if box == None:
                os.remove(tmp_dir)
                #print("Removed:",filename)
            else:
                borderSize=0
                if box[2] > box[3]:
                    transform = transforms.Compose([transforms.CenterCrop(box[3])])
                    borderSize = math.ceil(box[3]*0.20)
                else:
                    transform = transforms.Compose([transforms.CenterCrop(box[2])])
                    borderSize = math.ceil(box[2]*0.20)
                img = transform(img)
                if(pad == 1):
                    if isinstance(borderSize, int) or isinstance(borderSize, tuple):
                        img = ImageOps.expand(img, border=borderSize, fill="white")
                    else:
                        raise runtimeerror('border is not an integer or tuple!')
                if box[2] > maxsize or box[3] > maxsize:
                    img = img.resize((maxsize, maxsize))
                rgb_im = img.convert('RGB')

                rgb_im.save(os.path.join(destination,classification + "." + str(count).zfill(5) + ".jpg"))
                #os.remove(os.path.join(source, filename))
                #print("Produced: " + classification + "." + str(count).zfill(5) + ".jpg")
                count = count + 1
    #print("RENAMING END")
                
def CroppingImages(source, destination, classification, maxsize,pad):
    #print("RENAMING START")
    files = os.listdir(destination)
    count = len(files)
    for filename in os.listdir(source):
        if filename.endswith(".PNG") or filename.endswith(".png") or filename.endswith(".JPG") or filename.endswith(".jpg"):
            tmp_dir = os.path.join(source,filename)
            img = Image.open(tmp_dir)
            box = img.getbbox()
            if box == None:
                os.remove(tmp_dir)
                #print("Removed:",filename)
            else:
                borderSize=0
                transform = transforms.Compose([transforms.CenterCrop(maxsize)])
                img = transform(img)
                if(pad == 1):
                    if isinstance(borderSize, int) or isinstance(borderSize, tuple):
                        img = ImageOps.expand(img, border=borderSize, fill="white")
                    else:
                        raise runtimeerror('border is not an integer or tuple!')
                if box[2] > maxsize or box[3] > maxsize:
                    img = img.resize((maxsize, maxsize))
                rgb_im = img.convert('RGB')

                rgb_im.save(os.path.join(destination,classification + "." + str(count).zfill(5) + ".jpg"))
                #os.remove(os.path.join(source, filename))
                #print("Produced: " + classification + "." + str(count).zfill(5) + ".jpg")
                count = count + 1
    #print("RENAMING END")


def compute_lbp(arr):
    """Find LBP of all pixels.
    Also perform Vectorization/Normalization to get feature vector.
    """
    lbp = local_binary_pattern(arr, n_points, radius, METHOD)
    lbp = lbp.ravel()
    # feature_len = int(lbp.max() + 1)
    feature = np.zeros(n_bins)
    for i in lbp:
        feature[int(i)] += 1
    feature /= np.linalg.norm(feature, ord=1)
    return feature

def load_data(tag=testfolder):
    """Load (training/test) data from the directory.
    Also do preprocessing to extra features. 
    """
    tag_dir = Path.cwd() / tag
    #print(tag_dir)
    vec = []
    cat = []
    for cat_dir in tag_dir.iterdir():
        cat_label = cat_dir.stem
        #print(cat_label)
        for img_path in cat_dir.glob('*.jpg'):
            img = Image.open(img_path.as_posix())
            #print(img_path.as_posix(), img.mode)
            if img.mode != 'L':
                img = ImageOps.grayscale(img)
                img.save(img_path.as_posix())
            arr = np.array(img)
            feature = compute_lbp(arr)
            vec.append(feature)
            cat.append(cat_label)
    return vec, cat



def get_conf_mat(y_pred, y_target, n_cats):
    #Build confusion matrix from scratch.
    #(This part could be a good student assignment.)
    
    conf_mat = np.zeros((n_cats, n_cats))
    n_samples = y_target.shape[0]
    for i in range(n_samples):
        _t = y_target[i]
        _p = y_pred[i]
        conf_mat[_t, _p] += 1
    norm = np.sum(conf_mat, axis=1, keepdims=True)
    return conf_mat / norm

def vis_conf_mat(conf_mat, cat_names, acc):
    #Visualize the confusion matrix and save the figure to disk.
    n_cats = conf_mat.shape[0]

    fig, ax = plt.subplots()
    # figsize=(10, 10)

    cmap = cm.Blues
    im = ax.matshow(conf_mat, cmap=cmap)
    im.set_clim(0, 1)
    ax.set_xlim(-0.5, n_cats - 0.5)
    ax.set_ylim(-0.5, n_cats - 0.5)
    ax.set_xticks(np.arange(n_cats))
    ax.set_yticks(np.arange(n_cats))
    ax.set_xticklabels(cat_names)
    ax.set_yticklabels(cat_names)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    for i in range(n_cats):
        for j in range(n_cats):
            text = ax.text(j, i, round(
                conf_mat[i, j], 2), ha="center", va="center", color="w")

    cbar = fig.colorbar(im)

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    _title = 'Normalized confusion matrix, acc={0:.2f}'.format(acc)
    ax.set_title(_title)

    #plt.show()
    #_filename = 'conf_mat.png'
    #plt.savefig(_filename, bbox_inches='tight')


    
if __name__ == '__main__':
    #sizes = [128, 192, 256, 320, 384, 448, 512]
    sizes = [384]
    Category = ["Healthy","Black Sigatoka"]
    accu = 0
    accuSize = 0
    #,"orig-B","testCC","testC1728","testC1664","testC1536","testC1024","testC512"]
    af = ["orig"]
    for z in af:
        print("Test Folder is ",z)
        for s in sizes:
            catIndex = 0
            image_maxsize = s
            pad = 0
            augmentFolder = z
            for i in Category:
                augment_folder = i
                classification_name = i
                
                #Get current directory of python file
                directory = os.path.dirname(os.path.realpath(__file__))
                SourceFolderDir = os.path.join(directory, augmentFolder)
                SourceFolderDir = os.path.join(SourceFolderDir, augment_folder)
                if(pad == 0):
                    RenameFolderDir = os.path.join(directory, 'test')
                    RenameFolderDir = os.path.join(RenameFolderDir, augment_folder)
                else:
                    RenameFolderDir = os.path.join(directory, augment_folder + "_Resized" + str(image_maxsize) + "_Padded")
                
                #Create Directories
                CreateDirectory(RenameFolderDir)
                
                #files = os.listdir(SourceFolderDir)
                #no_of_files = len(files)
                RenamingImages(SourceFolderDir, RenameFolderDir, classification_name, image_maxsize, pad)
            vec_train = np.load("vec_train.npy")
            cat_train = np.load("cat_train.npy")
            le = preprocessing.LabelEncoder()
            le.fit(cat_train)
            label_train = le.transform(cat_train)

            vec_test, cat_test = load_data(testfolder)              # load test data
            label_test = le.transform(cat_test)


            # SVM
            clf = LinearSVC(random_state=0, tol=1e-5)
            clf.fit(vec_train, label_train)             # fit SVM using training data

            pick = open('texture.p','wb')
            pickle.dump(clf,pick)
            pick.close()

            # evaluation
            prediction = clf.predict(vec_test)          # make prediction on the test data
            print(prediction)
            # visualization
            cmat = get_conf_mat(y_pred=prediction, y_target=label_test,n_cats=len(le.classes_))
            acc = cmat.trace() / cmat.shape[0]
            print("Size ", s, "Accuracy", acc)
            #vis_conf_mat(cmat, le.classes_, acc)
            if(acc >= accu):
                accu = acc
                accusize = s
    print("Highest accuracy is ", accu)#, " at size ",accusize)


