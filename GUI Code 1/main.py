from picamera import PiCamera
from picamera import Color
import time


def take_picture():
    camera.capture(output)
    camera.stop_preview()

if __name__ == '__main__':
    camera.start_preview()
    camera.annotate_size = 120 
    camera.annotate_foreground = Color('black')
    camera.annotate_background = Color('white')
    camera.annotate_text = " I am what I am " 
    time.sleep(1)
    camera.capture("capture.jpg")
    camera.stop_preview()
