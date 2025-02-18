import onnxruntime as ort
import numpy as np
import time

model_path = '/Users/layvvs/Desktop/Studying/BMSTU/Diploma-2024-2025/code/vision-sense-net/models/yolov8n-face.onnx'
session = ort.InferenceSession(model_path)

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

input_data = np.random.randn(*input_shape).astype(np.float32)

for _ in range(100):
    session.run(None, {input_name: input_data})

num_runs = 1000
start_time = time.time()
for _ in range(num_runs):
    session.run(None, {input_name: input_data})
end_time = time.time()

avg_latency = (end_time - start_time) / num_runs
print(f"Average latency: {avg_latency * 1000:.2f} ms")
print(f"FPS: {1 / avg_latency:.2f}")
