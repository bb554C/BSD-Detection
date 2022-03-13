import cv2
import numpy as np
 
onnx_model_path = "./BSD_model.onnx"
sample_image = "./test/38.jpg"
 
#The Magic:
net =  cv2.dnn.readNetFromONNX(onnx_model_path) 
image = cv2.imread(sample_image)
image = cv2.resize(image, (256, 256))

blob = cv2.dnn.blobFromImage(image, 1, (256, 256),(0, 0, 0), swapRB=True,)
net.setInput(blob)
preds = net.forward()
biggest_pred_index = np.array(preds)[0].argmax()
print ("Predicted class:",biggest_pred_index)
 
categories = ["cat","dog"]
print("The class",biggest_pred_index, "correspond to", categories[biggest_pred_index])
