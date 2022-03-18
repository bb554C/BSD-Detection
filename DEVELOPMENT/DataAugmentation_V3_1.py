import Augmentor
import os
import time
import random
import shutil
from PIL import Image
import torchvision.transforms as transforms


#This code allows us to augmetn the image dataset in batches.
#This reduces the issues of computer crashing when augmenting large images and large datasets


#User-Defined Variables
augment_folder = "Raw-BSD"
file_count_limit = 5
multiplier = 30
image_maxsize = 1024
classification_name = "BlackSigatoka"

#Get current directory of python file
directory = os.path.dirname(os.path.realpath(__file__))

#Set all directories needed by the augmentor
dest_folder = augment_folder + "_Augmented"
comp_folder = augment_folder + "_Original"
source_folder = directory + "\\" + augment_folder
dest_augmented_folder = directory + "\\" + dest_folder
dest_augmented_folder_output = dest_augmented_folder + "\\" + "output\\"
completed_folder = directory + "\\" + comp_folder

#Create Augmented Folder
try:
    os.mkdir(dest_augmented_folder)
    print("Folder Created: ",dest_folder)
except:
    print("Folder Already Exist.",dest_folder)

try:
    os.mkdir(completed_folder)
    print("Folder Created: ",comp_folder)
except:
    print("Folder Already Exist.",comp_folder)

files = os.listdir(source_folder)
no_of_files = len(files)

while no_of_files > 0:
    files = os.listdir(source_folder)
    no_of_files = len(files)
    if no_of_files < file_count_limit:
        cur_file_limit = no_of_files
    else:
        cur_file_limit = file_count_limit
    #Transfer a small random sample of images to augmented folder
    for file_name in random.sample(files, cur_file_limit):
        shutil.move(os.path.join(source_folder, file_name), dest_augmented_folder)
    count = 0
    for filename in os.listdir(dest_augmented_folder):
        if filename.endswith("PNG") or filename.endswith("png") or filename.endswith("JPG") or filename.endswith("jpg"):
            count = count + 1
    if cur_file_limit != 0:
        p = Augmentor.Pipeline(dest_folder,save_format="PNG")
        p.skew_tilt(probability=0.75, magnitude=0.3)
        p.skew_corner(probability=0.75, magnitude=0.3)
        p.rotate_random_90(probability=0.75)
        p.rotate(probability=0.75, max_left_rotation=20, max_right_rotation=20)
        p.shear(probability=0.75, max_shear_left=18, max_shear_right=18)
        p.flip_random(probability=0.75)
        p.random_brightness(probability=0.75, min_factor=0.9, max_factor=1.1)
        p.random_contrast(probability=0.75, min_factor=0.80, max_factor=0.90)
        p.sample(count*multiplier)
    for filename in os.listdir(dest_augmented_folder):
        if filename.endswith(".PNG") or filename.endswith("png") or filename.endswith("JPG") or filename.endswith("jpg"):
            shutil.copy(os.path.join(dest_augmented_folder, filename), dest_augmented_folder_output)
            shutil.move(os.path.join(dest_augmented_folder, filename), completed_folder)
    time.sleep(1)

count = 0
#Checking for bad images
for filename in os.listdir(dest_augmented_folder_output):
    if filename.endswith("PNG") or filename.endswith("png") or filename.endswith("JPG") or filename.endswith("jpg"):
        tmp = dest_augmented_folder_output + filename
        img = Image.open(tmp)
        box = img.getbbox()
        if box == None:
            os.remove(tmp)
            print("Removed:",filename)
        else:
            #Renaming images to yout naming standard
            if box[2] > box[3]:
                transform = transforms.Compose([transforms.CenterCrop(box[3])])
            else:
                transform = transforms.Compose([transforms.CenterCrop(box[2])])
            img = transform(img)
            if box[2] > image_maxsize or box[3] > image_maxsize:
                img = img.resize((image_maxsize , image_maxsize))
            rgb_im = img.convert('RGB')
            rgb_im.save(dest_augmented_folder + "\\" + "Healthy." + str(count) + ".jpg")
            print("Produced: " + classification_name + "." + str(count) + ".jpg")
            count = count + 1
print("COMPLETED: All augmented files are found in Raw-Healthy_Augmented")

#Return Raw imges back to source folder
for filename in os.listdir(completed_folder):
    if filename.endswith(".PNG") or filename.endswith("png") or filename.endswith("JPG") or filename.endswith("jpg"):
        shutil.move(os.path.join(completed_folder, filename), source_folder)
for filename in os.listdir(dest_augmented_folder_output):
    if filename.endswith(".PNG") or filename.endswith("png") or filename.endswith("JPG") or filename.endswith("jpg"):
        os.remove(os.path.join(dest_augmented_folder_output, filename))
        
#Remove TempDirectories
os.rmdir(completed_folder) 
os.rmdir(dest_augmented_folder_output) 
