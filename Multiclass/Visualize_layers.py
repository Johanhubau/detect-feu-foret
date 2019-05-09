import keras
from keras import Model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
# Here you load your trained network
path_model="/home/alexis51151/detect-feu-foret/Models/MulticlassV1"
model = keras.models.load_model(path_model)

layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input,outputs=layer_outputs)

# Here you load your image you want to visualize when passed through the neural network
image_path = "/home/alexis51151/detect-feu-foret/Internet/feu12.jpg"
img = Image.open(image_path)
img = img.resize((256,256))
img = np.array(img)
img = img.reshape((1,256,256,3))
activations = activation_model.predict(img)

def display_activation(activations, col_size, row_size, act_index):
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size,squeeze=False, figsize=(row_size*2.5, col_size*1.5))
    for row in range(0, row_size):
        for col in range(0, col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1

display_activation(activations, 4, 4, 5)
plt.show()