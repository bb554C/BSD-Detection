import torch as t
import torch.onnx
from ShuffleNet2 import ShuffleNet2

def Convert_ONNX(): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = t.randn(2, 3, 256, 256, requires_grad=True)  

    # Export the model   
    t.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "BSD_model.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True  # whether to execute constant folding for optimization 
         ) 
    print(" ") 
    print('Model has been converted to ONNX')


if __name__ == "__main__": 
    num_classes = 2
    input_size = 256
    net_type = 1
    # Let's build our model 
    #train(5) 
    #print('Finished Training') 

    # Test which classes performed well 
    #testAccuracy() 

    # Let's load the model we just created and test the accuracy per label 
    model = ShuffleNet2(num_classes, input_size, net_type)
    path = "BSD_model.pkl" 
    model.load_state_dict(torch.load(path)) 

    # Test with batch of images 
    #testBatch() 
    # Test how the classes performed 
    #testClassess() 
 
    # Conversion to ONNX 
    Convert_ONNX() 
