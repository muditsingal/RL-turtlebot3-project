import numpy as np
#import tensorflow as tf
#import tflite_runtime as tf
from tflite_runtime.interpreter import Interpreter

# Loading the TensorFlow Lite model
interpreter =Interpreter(model_path='/home/rpi/models/lite_model.tflite')
interpreter.allocate_tensors()

# Get the input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#print(input_details)
#print(output_details)

# Create a numpy array to hold the input data
#input_data = np.zeros(input_details[0]['shape'], dtype=np.float32)
print("Req shape: ", input_details[0]['shape'])
input_data = np.random.rand(20).astype(np.float32).reshape(input_details[0]['shape'])
#print("Curr inp shape", input_data.shape)

# Run the inference
print(input_details[0])
print(input_details[0]['index'])
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get the output data
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Output data:", output_data[0])
