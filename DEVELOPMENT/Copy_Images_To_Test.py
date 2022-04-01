import os
import shutil
import random

sourceFolder = "Raw-BSD_Resized256"
trainFolder = "BlackSigatoka"

directory = os.path.dirname(os.path.realpath(__file__))
source = os.path.join("D:\\","OneDrive - Mapúa University", "CPE200-2L_E01_Baguisi-Buenaventura", "Image Dataset", sourceFolder)
testdestination = os.path.join(directory,"test", trainFolder)
for i in range(1):
    filename = random.choice([
        x for x in os.listdir(source)
        if os.path.isfile(os.path.join(source, x))
    ])
    shutil.copy(os.path.join(source, filename), os.path.join(testdestination, filename))
    print("Image", filename, "copied")
