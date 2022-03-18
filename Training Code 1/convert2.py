import torch as t
import torch.onnx
from ShuffleNet2 import ShuffleNet2

if __name__ == "__main__": 
    num_classes = 2
    input_size = 256
    net_type = 1
    model = ShuffleNet2(num_classes, input_size, net_type)
    path = "BSD_Model.pkl"
    model.load_state_dict(t.load(path))
    model.eval() 
    dummy_input = t.randn(32, 3, 256, 256, requires_grad=True)  
    t.onnx.export(model,
         dummy_input,
         "BSD_Model.onnx",
         export_params=False,
         opset_version=10,
         do_constant_folding=False
         ) 
    print('Model has been converted to ONNX')
