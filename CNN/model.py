"""
CNN Model Creation Script

Author: [Your Name]
Created: [Date]

Description:
This script defines a function to create a Convolutional Neural Network (CNN) model using TensorFlow's Keras API. The model architecture consists of a series of Conv3D layers with ReLU activation, followed by a skip connection to incorporate the input data. The model is compiled with the Adam optimizer and mean squared error loss.

Libraries Used:
- tensorflow.keras.models.Model
- tensorflow.keras.layers.*
- tensorflow.keras.initializers.HeNormal
- tensorflow.keras.optimizers.Adam

Usage:
- Modify the input_shape, num_filters, filter_size, and num_blocks according to your specific task.
- Execute the script using the Python interpreter.
- The script will create and print a summary of the CNN model.

Note:
- Ensure that TensorFlow is installed before running the script.
- Adjust the script parameters and configurations as needed.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam

# Could also pass 2 inputs seperatly and concatenate as first layer
def create_model(input_shape, num_blocks = 10, num_filters = 64, filter_size = [3, 3, 3]):
    inputs = Input(shape=input_shape)
    # Input should contain phs and magn image in channels
    # Add `num_blocks` of Conv3D + ReLU
    x = Conv3D(num_filters, filter_size, padding="same", activation= 'relu', kernel_initializer=HeNormal())(inputs)
    for _ in range(num_blocks-1):
        x = Conv3D(num_filters, filter_size, padding="same", activation= 'relu', kernel_initializer=HeNormal())(x)
    x = Conv3D(1, [3,3,3], padding="same", kernel_initializer=HeNormal())(x) # Not sure about filter_size, activation = linear ?
    x = Add()([x[...,0], inputs[..., 0]]) # Skip connection between interpolated low res phs image and conv/relu output
    model = Model(inputs,x)
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mean_squared_error'])

    return model

if __name__ == "__main__":
    # Example usage
    input_shape = (64, 32, 32, 32, 2)  # Example input shape (32x32x32) with 2 channels
    num_filters = 64  # Number of convolution filters
    filter_size = (3, 3, 3)  # Size of convolution filters
    num_blocks = 10  # Number of convolution blocks

    model = create_model(input_shape)
    model.summary()

