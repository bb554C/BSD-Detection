from PIL import Image
import os
import torchvision.transforms as transforms

directory = 'newtest/'

count = 0
maxsize = 512
for filename in os.listdir(directory):
    if filename.endswith(".PNG"):
        tmp = directory + filename
        img = Image.open(tmp)
        box = img.getbbox()
        if box[2] >= box[3]:
            transform = transforms.Compose([transforms.CenterCrop(box[3])])
        else:
            transform = transforms.Compose([transforms.CenterCrop(box[2])])
        img = transform(img)
        if box[2] > maxsize or box[3] > maxsize
        img = img.resize((maxsize , maxsize))
        rgb_im = img.convert('RGB')
        rgb_im.save(directory + "output/" + "Healthy." + str(count) + ".jpg")
        print("Produced: " + "Healthy." + str(count) + ".jpg")
        count = count + 1
