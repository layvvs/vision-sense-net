from .net import ONNXNet
from enum import Enum
import numpy as np


RESNET_INPUT_SIZE = 224


class Emotions(Enum):
    angry = 0
    happy = 1
    neutral = 2
    sad = 3
    surprize = 4


def softmax(x):
    x_exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x_exp / np.sum(x_exp, axis=-1, keepdims=True)


class ResNet(ONNXNet):
    model_name = 'resnet50'

    def preprocess(self, frame):
        resized_frame = frame.resize((RESNET_INPUT_SIZE, RESNET_INPUT_SIZE))
        processed_frame = np.array(resized_frame).transpose(2, 0, 1).astype(np.float32) / 255
        return np.expand_dims(processed_frame, axis=0)

    def run(self, frame):
        result = self._run(self.preprocess(frame))
        emotion_class = Emotions(np.argmax(softmax(result))).value
        return emotion_class
