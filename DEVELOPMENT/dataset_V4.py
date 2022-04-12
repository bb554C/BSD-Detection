from torchvision import datasets, transforms
import os
import time
from torch.utils import data
import numpy as np
from PIL import Image
import copy

class BSD(data.Dataset):
  def __init__(self, root, trans=None, train=True, test=False):
    self.test = test
    self.train = train
    imgs = [os.path.join(root, img) for img in os.listdir(root)]
    if test:
      sorted(imgs, key=lambda x: int(x.split(".")[-2].split("/")[-1])) 
    else:
      sorted(imgs, key=lambda x: int(x.split(".")[-2])) 
    # shuffle
    np.random.seed(100)
    imgs = np.random.permutation(imgs)
    # split dataset
    if self.test:
      self.imgs = imgs
    elif train:
      self.imgs = imgs[:int(0.7*len(imgs))]
    else:
      self.imgs = imgs[int(0.7*len(imgs)):]
    if trans==None:
      self.trans = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.ToTensor()
                                        ])
        
  def __getitem__(self, index):
    imgpath = self.imgs[index]
    label = 2
    if self.test:
      label = int(imgpath.split(".")[-2].split("/")[-1])
    else:
      kind = imgpath.split("/train\\")[-1]
      kind = kind.split(".")
      if kind[0] == "Healthy":
        label = 0
      elif kind[0] == "BlackSigatoka":
        label = 1
      elif kind[0] == "Background":
        label = 2
      #print(kind[0], label)
      if kind[0] != "BlackSigatoka" and kind[0] != "Healthy" and kind[0] != "Background":
        print("ERROR: Does not recognize the dataset. Please check directory path on dataset.py")
    img = Image.open(imgpath)
    img = self.trans(img)
    return img, label
  
  def __len__(self):
    return len(self.imgs)
