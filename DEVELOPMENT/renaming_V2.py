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

def CreateDirectory(dir):
    try:
        os.mkdir(dir)
        print("Folder Created: ",dir)
    except:
        print("Folder Already Exist.",dir)

def RenamingImages(source, destination, classification, maxsize,pad):
    print("RENAMING START")
    files = os.listdir(destination)
    count = len(files)
    for filename in os.listdir(source):
        if filename.endswith(".PNG") or filename.endswith(".png") or filename.endswith(".JPG") or filename.endswith(".jpg"):
            tmp_dir = os.path.join(source,filename)
            img = Image.open(tmp_dir)
            box = img.getbbox()
            if box == None:
                os.remove(tmp_dir)
                print("Removed:",filename)
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
                print("Produced: " + classification + "." + str(count).zfill(5) + ".jpg")
                count = count + 1
    print("RENAMING END")


    
if __name__ == '__main__':
    augment_folder = "Raw-BSD_Resized256"
    classification_name = "BlackSigatoka"
    image_maxsize = 256
    pad = 0
    
    #Get current directory of python file
    directory = os.path.dirname(os.path.realpath(__file__))
    
    SourceFolderDir = os.path.join(directory, augment_folder)
    RenameFolderDir = os.path.join(directory, augment_folder + "_Resized" + str(image_maxsize))
    
    #Create Directories
    CreateDirectory(RenameFolderDir)
    
    files = os.listdir(SourceFolderDir)
    no_of_files = len(files)
    RenamingImages(SourceFolderDir, RenameFolderDir, classification_name, image_maxsize, pad)
    





