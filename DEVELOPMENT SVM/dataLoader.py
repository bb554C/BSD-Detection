from pathlib import Path
import cv2
import numpy as np
import os
import shutil
from skimage.feature import local_binary_pattern

def CreateDirectory(dir):
    try:
        os.mkdir(dir)
        print("Folder Created: ",dir)
    except:
        print("Folder Already Exist.",dir)

def RemoveDirectories(dir):
    try:
        files = os.listdir(dir)
        no_of_files = len(files)
        if no_of_files == 0:
            os.rmdir(dir)
        else:
            print("WARNING: Directory still containes files")
            print(dir)
    except:
        print("WARNING: Directory does not exist")
        print(dir)

def MoveFiles(source, destination):
    for filename in os.listdir(source):
        shutil.move(os.path.join(source, filename), destination)

def compute_lbp(img):
    radius = 2
    n_points = 8 * radius
    METHOD = 'uniform'
    n_bins = n_points + 2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(img, n_points, radius, METHOD)
    lbp = lbp.ravel()
    feature = np.zeros(n_bins)
    for i in lbp:
        feature[int(i)] += 1
    feature /= np.linalg.norm(feature, ord=1)
    return feature.flatten()

def compute_colorHist(img):
    bins = (8, 8, 8)
    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    eq_V = cv2.equalizeHist(V)
    eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2BGR)
    hist = cv2.calcHist([eq_image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def compute_grayHist(img):
    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    eq_V = cv2.equalizeHist(V)
    eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2BGR)
    image = cv2.cvtColor(eq_image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([image], [0], None, [8], [0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def load_data(tag,maxsize):
    tag_dir = Path.cwd() / tag
    print(tag_dir)
    data, labels = [], []
    for cat_dir in tag_dir.iterdir():
        label = cat_dir.stem
        for img_path in cat_dir.glob('*.jpg'):
            #print(img_path.as_posix())
            image = cv2.imread(img_path.as_posix())
            image = cv2.resize(image,(maxsize,maxsize))
            #feature =  compute_lbp(image)
            #feature = compute_grayHist(image)
            #feature = compute_colorHist(image)
            #BEST BELOW 93.33 colorHist bin 8 colorGray bin 8 lbp rad 2
            feature = np.concatenate([compute_colorHist(image), compute_grayHist(image), compute_lbp(image)])
            data.append(feature)
            labels.append(label)
    return data, labels

def get_conf_mat(y_pred, y_target, n_cats):
    conf_mat = np.zeros((n_cats, n_cats))
    n_samples = y_target.shape[0]
    for i in range(n_samples):
        _t = y_target[i]
        _p = y_pred[i]
        conf_mat[_t, _p] += 1
    norm = np.sum(conf_mat, axis=1, keepdims=True)
    return conf_mat / norm

def vis_conf_mat(conf_mat, cat_names, acc):
    n_cats = conf_mat.shape[0]

    fig, ax = plt.subplots()

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

    plt.show()
    _filename = 'conf_mat.png'
    #plt.savefig(_filename, bbox_inches='tight')
