from guizero import App, Text, PushButton, Picture

def exit():
    if app.yesno("Close", "Do you want to quit?"):
        app.destroy()

if __name__ == '__main__':
    app = App("Black Sigatoka", 800, 800)
    captured_photo = "capture.jpg"
    message = Text(app, "BLACK SIGATOKA DETECTOR")
    output = Picture(app, captured_photo, width=300, height=300)
    button = PushButton(app, command=exit, text="Exit Application")
    app.set_full_screen()
    app.display()
