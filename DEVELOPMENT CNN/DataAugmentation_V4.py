import Augmentor
import os
import random
import shutil
from PIL import Image
from PIL import ImageFile
import torchvision.transforms as transforms
import threading
import time
import math
ImageFile.LOAD_TRUNCATED_IMAGES = True

#This code allows us to augmetn the image dataset in batches.
#This reduces the issues of computer crashing when augmenting large images and large datasets

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
    
def MoveImages(source, destination):
    for filename in os.listdir(source):
        if filename.endswith(".PNG") or filename.endswith(".png") or filename.endswith(".JPG") or filename.endswith(".jpg"):
            shutil.move(os.path.join(source, filename), destination)

def MoveCopyImages(source, destination, copy_destination):
    for filename in os.listdir(source):
        if filename.endswith(".PNG") or filename.endswith(".png") or filename.endswith(".JPG") or filename.endswith(".jpg"):
            shutil.copy(os.path.join(source, filename), copy_destination)
            shutil.move(os.path.join(source, filename), destination)
            
def MoveImagesRandomLimited(source, destination, limit, files):
    for filename in random.sample(files, limit):
        if filename.endswith(".PNG") or filename.endswith(".png") or filename.endswith(".JPG") or filename.endswith(".jpg"):
            shutil.move(os.path.join(source, filename), destination)
            
def AugmentorStart(source, count, multiplier, limit):
    if limit != 0:
        p = Augmentor.Pipeline(source,save_format="PNG")
        p.skew_tilt(probability=0.75, magnitude=0.3)
        p.skew_corner(probability=0.75, magnitude=0.3)
        p.rotate_random_90(probability=0.75)
        p.rotate(probability=0.75, max_left_rotation=20, max_right_rotation=20)
        p.shear(probability=0.75, max_shear_left=18, max_shear_right=18)
        p.flip_random(probability=0.75)
        p.random_brightness(probability=0.75, min_factor=0.9, max_factor=1.1)
        p.random_contrast(probability=0.75, min_factor=0.80, max_factor=0.90)
        p.sample(count*multiplier)

def RenamingImages(source, destination, classification, maxsize):
    print("RENAMING THREAD START")
    files = os.listdir(destination)
    count = len(files)
    x = 0
    for filename in os.listdir(source):
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
            rgb_im.save(os.path.join(destination,classification + "." + str(count).zfill(5) + ".jpg"))
            os.remove(os.path.join(source, filename))
            print("Produced: " + classification_name + "." + str(count).zfill(5) + ".jpg")
            count = count + 1
    print("RENAMING THREAD END")
                
if __name__ == '__main__':
#User-Defined Variables
    augment_folder = "Raw-Background_Resized512"
    classification_name = "Unknown"
    datasetgoal = 2000
    image_maxsize = 256
    #Get current directory of python file
    directory = os.path.dirname(os.path.realpath(__file__))

    #Set all directories needed by the augmentor
    SourceFolderDir = os.path.join(directory, augment_folder) 
    AugmentFolderDir = os.path.join(directory,augment_folder + "_AugmentTEMP") #temp
    ProcessedFolderDir = os.path.join(directory, augment_folder + "_ProcessTEMP") #temp
    RenameFolderDir = os.path.join(directory, augment_folder + "_RenameTEMP") #temp
    AugmentorFolderDir = os.path.join(directory,augment_folder + "_AugmentTEMP","output") #temp
    DestinationFolderDir = os.path.join(directory, augment_folder + "_Augmented" + str(image_maxsize))

    #Create Directories
    CreateDirectory(AugmentFolderDir)
    CreateDirectory(ProcessedFolderDir)
    CreateDirectory(RenameFolderDir)
    CreateDirectory(DestinationFolderDir)

    files = os.listdir(SourceFolderDir)
    no_of_files = len(files)
    multiplier = math.ceil(datasetgoal / no_of_files)
    print("multiplier = ", multiplier)
    file_count_limit = math.ceil(50 / multiplier)
    while no_of_files > 0:
        files = os.listdir(SourceFolderDir)
        no_of_files = len(files)
        if no_of_files < file_count_limit:
            cur_file_limit = no_of_files
        else:
            cur_file_limit = file_count_limit
        MoveImagesRandomLimited(SourceFolderDir, AugmentFolderDir, cur_file_limit, files)
        AugmentorStart(AugmentFolderDir, cur_file_limit, multiplier, cur_file_limit)
        for thread in threading.enumerate():
            if thread.name == "RenamingThread":
                print("waiting to join")
                thread.join()
        MoveCopyImages(AugmentFolderDir, ProcessedFolderDir, AugmentorFolderDir)
        MoveImages(AugmentorFolderDir, RenameFolderDir)
        if cur_file_limit != 0:
            tr = threading.Thread(target=RenamingImages,args=(RenameFolderDir, DestinationFolderDir, classification_name, image_maxsize))
            tr.name = "RenamingThread"
            tr.start()
    tr.join()
    #Move Images to their final folders
    MoveImages(ProcessedFolderDir, SourceFolderDir)
    time.sleep(1)    
    #DeleteDirectories
    RemoveDirectories(AugmentorFolderDir)
    RemoveDirectories(AugmentFolderDir)
    RemoveDirectories(ProcessedFolderDir)
    RemoveDirectories(RenameFolderDir)
    print("AUGMENTATION COMPLETED")
    print("multiplier = ", multiplier)
