from PIL import Image, ImageTk
from picamera import PiCamera
import tkinter as tk
import threading as thr
import time
import os
from ShuffleNet2 import ShuffleNet2
from torchvision import transforms
import torchvision
import torch
from io import BytesIO



def detect_image_class(model, pic):
    preprocess = transforms.Compose([transforms.Resize(256),
                                     transforms.ToTensor()])
    image = preprocess(pic)
    input_batch = image.unsqueeze(0)
    input_batch = input_batch.to('cpu')
    model.to('cpu')
    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    categories = ["Healthy", "BlackSigatoka", "Unkown"]
    top_prob, top_id = torch.topk(probabilities, 1) 
    return categories[top_id[0]]

def update_GUI(image_display, classification):
    canvas.itemconfig(image_canvas, image = image_display)
    text_output.config(text = classification)

def update_image(directory, cam, model):
    stop = 1
    while stop != 0:
        imgPath = BytesIO()
        cam.capture(imgPath, format='jpeg')
        imgPath.seek(0)
        image_temp = Image.open(imgPath).convert('RGB')
        image_temp = image_temp.resize((256, 256), Image.ANTIALIAS)
        classification = detect_image_class(model, image_temp)
        img_display = ImageTk.PhotoImage(image_temp)
        th1 = thr.Thread(target=update_GUI,arge=(image_display, classification))
        th1.start()

def multithread():
    #Setup Directories
    directory = os.path.dirname(os.path.realpath(__file__))
    #SetupCamera
    camera = PiCamera()
    camera.resolution = (1024, 1024)
    #Setup Model
    model = ShuffleNet2(2, 256, 2)
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            modelDir = os.path.join(directory, filename)
            model.load_state_dict(torch.load(modelDir, map_location=torch.device('cpu')))
    model.eval()
    thread = thr.Thread(target=update_image, args=(directory, camera, model))
    thread.daemon = True
    thread.start()
    
def exit_app():
    app.destroy()

if __name__ == '__main__':
    app = tk.Tk()
    app.attributes('-fullscreen', True)
    canvas = tk.Canvas(app, width=256, height=256,bg='black')
    canvas.pack()
    directory = os.path.dirname(os.path.realpath(__file__))
    imgPath = os.path.join(directory, "placeholder.jpg")
    
    image_temp = Image.open(imgPath).convert('RGB')
    image_temp = image_temp.resize((256, 256), Image.ANTIALIAS)
    img_display = ImageTk.PhotoImage(image_temp)
    image_canvas = canvas.create_image(0, 0, image = img_display, anchor = tk.NW)
    
    text_output = tk.Label(app, text = "NONE")
    text_output.pack()
    
    button_Exit = tk.Button(app, text="EXIT APPLICATION", command=exit_app)
    button_Exit.pack()
    
    app.after(1000, multithread)
    app.mainloop()
