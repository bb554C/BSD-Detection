import torch as t
import torch.onnx
from ShuffleNet2 import ShuffleNet2

def Convert_ONNX(): 
    model.eval() 
    dummy_input = t.randn(32, 3, 256, 256, requires_grad=True)  
    t.onnx.export(model,
         dummy_input,
         "BSD_Model.onnx",
         export_params=True,
         opset_version=10,
         do_constant_folding=True
         ) 
    print(" ") 
    print('Model has been converted to ONNX')


if __name__ == "__main__": 
    num_classes = 2
    input_size = 256
    net_type = 1
    model = ShuffleNet2()
    path = "BSD_Model.pkl" 
    model.load_state_dict(t.load(path)) 
    Convert_ONNX() 
