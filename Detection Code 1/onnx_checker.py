import onnx
onnx_model = onnx.load("./BSD_model.onnx")
onnx.checker.check_model(onnx_model)
