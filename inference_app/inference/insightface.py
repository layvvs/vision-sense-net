from .net import ONNXNet
import numpy as np


MAGFACE_INPUT_SIZE = 112


class InsightFace(ONNXNet):
    model_name = 'magface'

    def preprocess(self, frame):
        resized_frame = frame.resize((MAGFACE_INPUT_SIZE, MAGFACE_INPUT_SIZE))
        processed_frame = np.array(resized_frame).transpose(2, 0, 1).astype(np.float32) / 255
        return np.expand_dims(processed_frame, axis=0)

    def run(self, frame):
        vector = self._run(self.preprocess(frame))[0][0]
        return vector
