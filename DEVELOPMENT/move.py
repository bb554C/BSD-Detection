import os
import shutil

directory = os.path.dirname(os.path.realpath(__file__))
source = os.path.join("D:\\","OneDrive - Mapúa University", "CPE200-2L_E01_Baguisi-Buenaventura", "Image Dataset","Unknown_NonAugmented")
testdestination = os.path.join(directory,"test", "Unknown")
traindestination = os.path.join(directory,"train")   
for filename in os.listdir(source):
    if filename.endswith(".PNG") or filename.endswith(".png") or filename.endswith(".JPG") or filename.endswith(".jpg"):
        #shutil.move(os.path.join(source, filename), os.path.join(destination, filename))
        shutil.copy(os.path.join(source, filename), os.path.join(traindestination, filename))
        shutil.copy(os.path.join(source, filename), os.path.join(testdestination, filename))
        print("COPIED", filename)

        
