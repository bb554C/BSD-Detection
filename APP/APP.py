from PIL import Image, ImageTk
import tkinter as tk
import threading as thr
import time

def update_image():
    for x in range(36):
        path = str(x) + ".jpg"
        image_temp = Image.open(path)
        image_temp = image_temp.resize((250, 250), Image.ANTIALIAS)
        img_display = ImageTk.PhotoImage(image_temp)
        canvas.itemconfig(image_canvas, image = img_display)
        text_output.config(text = str(x))
        time.sleep(0.5)
        
def multithread():
    thread = thr.Thread(target=update_image)
    thread.daemon = True
    thread.start()
    
def exit_app():
    app.destroy()

if __name__ == '__main__':
    app = tk.Tk()
    app.attributes('-fullscreen', True)
    canvas = tk.Canvas(app, width=250, height=250,bg='black')
    canvas.pack()

    image_temp = Image.open("2.jpg")
    image_temp = image_temp.resize((250, 250), Image.ANTIALIAS)
    img_display = ImageTk.PhotoImage(image_temp)
    image_canvas = canvas.create_image(0, 0, image = img_display, anchor = tk.NW)
    
    text_output = tk.Label(app, text = "NONE")
    text_output.pack()
    
    button_Exit = tk.Button(app, text="EXIT APPLICATION", command=exit_app)
    button_Exit.pack()
    
    app.after(1000, multithread)
    app.mainloop()
