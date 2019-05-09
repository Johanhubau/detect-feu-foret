#A function named classes_from_image whose purpose is to output the classes results from our neural network

import keras
from keras import Model
from PIL import Image
import numpy as np



# Here you load your image (must be at least 256 pixels in height and width to work with our neural network format)
image_path = "/home/alexis51151/detect-feu-foret/Internet/feu12.jpg"

# Here you load your trained network
path_model="/home/alexis51151/detect-feu-foret/Models/MulticlassV3"
model = keras.models.load_model(path_model)

# Your classes
classes = ["Feu", "Fumée ou brouillard", "Non feu", "Objet rouge"]

def classes_printing(array):
    for i in range(len(array[0])):
        print("Classe " + classes[i] + " à " + str(array[0][i]*100)+"%")
# Main fonction to ue ; input = path to your image ; output = printing of the prediction and an array
def classes_from_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))
    img = np.array(img)
    img = img.reshape((1, 256, 256, 3))
    res = model.predict(img)
    classes_printing(res)
    return(res[0])

test = classes_from_image(image_path)


