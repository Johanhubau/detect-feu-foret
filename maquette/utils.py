#A function named classes_from_image whose purpose is to output the classes results from our neural network

import keras
from keras import Model
from PIL import Image
import numpy as np
import random as rdm
from shutil import copyfile
import glob
import time

path_model="/home/heisenberg/gate/models/MulticlassV3"
model = keras.models.load_model(path_model)


MAIN_PATH = "/home/heisenberg/gate/DB/Bdd_perso/Test/"

class_names = ["Feu", "Fumée ou brouillard", "Non feu", "Objet rouge"]
class_folder_names = ["Fire/", "Fog/", "Not_fire/", "Red_object/"]

def classes_printing(array):
    for i in range(len(array[0])):
        print("Classe " + class_names[i] + " à " + str(array[0][i]*100)+"%")

def classes_write(array, text):
    file = open("net.txt", "w")
    for i in range(len(array)):
        file.write(str(array[i]))
        file.write("\n")
    file.write(str(text)+"\n")
    file.close()
# Main fonction to ue ; input = path to your image ; output = printing of the prediction and an array
def classes_from_image(image_path, model):
    img = Image.open(image_path)
    img = img.resize((256, 256))
    img = np.array(img)
    img = img.reshape((1, 256, 256, 3))
    res = model.predict(img)
    classes_printing(res)
    #
    # y_feature = np.zeros(shape=(1, 8, 8, 512))
    # y_feature[0] = conv_base.predict(img)
    # y_feature = np.reshape(y_feature, (1, 8*8*512))
    # y_prob = model.predict(y_feature)
    return(res[0])



while True:
    #Choose random image from random class
    choose_class = rdm.randint(0,3)
    image_path = MAIN_PATH + class_folder_names[choose_class]
    files = glob.glob(image_path + "*")
    chosen_image = rdm.randint(0, len(files)-1)
    image_path = files[chosen_image]

    classes = classes_from_image(image_path, model)

    classes_write(classes, choose_class)

    #Copy image
    copyfile(image_path, "/home/heisenberg/gate/detect-feu-foret/maquette/static/img.jpg")
    time.sleep(10)
