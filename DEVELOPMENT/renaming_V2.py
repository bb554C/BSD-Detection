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

def CreateDirectory(dir):
    try:
        os.mkdir(dir)
        print("Folder Created: ",dir)
    except:
        print("Folder Already Exist.",dir)

def RenamingImages(source, destination, classification, maxsize):
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
                if box[2] > box[3]:
                    transform = transforms.Compose([transforms.CenterCrop(box[3])])
                else:
                    transform = transforms.Compose([transforms.CenterCrop(box[2])])
                img = transform(img)
                if box[2] > maxsize or box[3] > maxsize:
                    img = img.resize((maxsize , maxsize))
                rgb_im = img.convert('RGB')
                rgb_im.save(os.path.join(destination,classification + "." + str(count) + ".jpg"))
                print("Produced: " + classification + "." + str(count) + ".jpg")
                count = count + 1
    print("RENAMING END")
    
if __name__ == '__main__':
    augment_folder = "Raw-BSD"
    classification_name = "testBSD"
    image_maxsize = 1024
    
    #Get current directory of python file
    directory = os.path.dirname(os.path.realpath(__file__))
    
    SourceFolderDir = os.path.join(directory, augment_folder)
    RenameFolderDir = os.path.join(directory, augment_folder + "_Renamed")
    
    #Create Directories
    CreateDirectory(RenameFolderDir)
    
    files = os.listdir(SourceFolderDir)
    no_of_files = len(files)
    RenamingImages(SourceFolderDir, RenameFolderDir, classification_name, image_maxsize)
    





