import os
import shutil

directory = os.path.dirname(os.path.realpath(__file__))
source = os.path.join(directory,"train")
kind = []
for filename in os.listdir(source):
    if filename.endswith(".PNG") or filename.endswith(".png") or filename.endswith(".JPG") or filename.endswith(".jpg"):
        kind = filename
        kind = kind.split(".")
        if kind[0] == "Unkown":
            os.remove(os.path.join(source, filename))
            print("REMOVED", filename)
