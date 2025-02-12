import torch
import onnx

model = torch.load('/Users/layvvs/Desktop/Studying/BMSTU/Diploma-2024-2025/code/vision-sense-net/models/fer_resnet50.pt', map_location=torch.device('cpu'))

model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

onnx_output_path = '/Users/layvvs/Desktop/Studying/BMSTU/Diploma-2024-2025/code/vision-sense-net/models/fer_resnet50.onnx'

torch.onnx.export(model, dummy_input, onnx_output_path, 
                  export_params=True,
                  do_constant_folding=True,
                  input_names=['images'],
                  output_names=['output'],
)


onnx_model = onnx.load(onnx_output_path)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")
