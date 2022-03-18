import os
import torch
import torchvision.transforms as transforms
from PIL import Image

directory = 'train/'

transform = transforms.Compose([
    transforms.ToTensor()
])
meantot, stdtot = 0 , 0
count = 0;
for filename in os.listdir(directory):
    count = count + 1
    tmp = directory + filename
    img = Image.open(tmp)
    img_tr = transform(img)
    mean = img_tr.mean([1,2])
    std = img_tr.std([1,2])
    meantot = meantot + mean
    stdtot = stdtot + std
    print(count)
print("Mean of the image:", meantot/count)
print("Std of the image:", stdtot/count)

