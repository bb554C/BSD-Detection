from PIL import Image
import os
import torchvision.transforms as transforms

#User-Defined Variables
dataset_folder = 'newtest'
image_maxsize = 1024
classification_name = "BlackSigatoka"

#Get current directory of python file
directory = os.path.dirname(os.path.realpath(__file__))
source_folder = directory + "\\" + dataset_folder
destination_folder = directory + "_Renamed"

#Create Augmented Folder
try:
    os.mkdir(destination_folder)
    print("Folder Created: ",dataset_folder + "_Renamed")
except:
    print("Folder Already Exist.", destination_folder)

count = 0
#Checking for bad images
for filename in os.listdir(source_folder):
    if filename.endswith("PNG") or filename.endswith("png") or filename.endswith("JPG") or filename.endswith("jpg"):
        tmp = os.path.join(source_folder , filename)
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
