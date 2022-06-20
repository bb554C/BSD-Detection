import os
import shutil
import random

sourceFolder = "Unknown_Augmented256"

directory = os.path.dirname(os.path.realpath(__file__))
source = os.path.join("D:\\","OneDrive - Mapúa University", "CPE200-2L_E01_Baguisi-Buenaventura", "Image Dataset", sourceFolder)
traindestination = os.path.join(directory,"train")   
for filename in os.listdir(source):
    if filename.endswith(".PNG") or filename.endswith(".png") or filename.endswith(".JPG") or filename.endswith(".jpg"):
        #shutil.move(os.path.join(source, filename), os.path.join(destination, filename))
        shutil.copy(os.path.join(source, filename), os.path.join(traindestination, filename))      
        print("COPIED", filename)
