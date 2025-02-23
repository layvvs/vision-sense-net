import onnxruntime as rt
import os


MODELS_PATH = f'{os.getcwd()}/models'


def find_model(model_name):
    return os.path.join(MODELS_PATH, f'{model_name}.onnx')


class ONNXNet:
    model_name = ''

    def __init__(self):
        model_path = find_model(self.model_name)
        self.model = rt.InferenceSession(model_path)
        self.input_name = self.model.get_inputs()[0].name

    def _run(self, frame):
        return self.model.run(None, {self.input_name: frame})
