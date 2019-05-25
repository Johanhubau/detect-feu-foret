import os
import numpy as np
import keras
from PIL import Image
# New imports
from keras.models import load_model
from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras import models
from keras import layers
from keras import optimizers
from matplotlib import pyplot as plt
import numpy as np

labels= {'Fire': 0, 'Fog': 2, 'Not_fire': 1, 'Red_object': 3, 'Smoke': 4} # Our neural network classes


# Here you load your trained network
path_model="/home/alexis51151/detect-feu-foret/Models/pretrained_VGG16_5classes"
#model = keras.models.load_model(path_model)
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
model = load_model("/home/alexis51151/Téléchargements/pretrained_VGG16_5classes")
model.summary()
# Here you load your image you want to visualize when passed through the neural network
path_image = "/home/alexis51151/Documents/Bdd_multiclasse/Test/"


# And then you launch this function to get the indicators
def F1(path_image: object) -> object: # Version pour VGG16_newVersion
    nb_classes=len(os.listdir(path_image))
    correctly_attributed=nb_classes*[0]
    attributed=nb_classes*[0]
    nb_elements=nb_classes*[0]
    confusion_matrix= np.zeros((nb_classes,nb_classes))
    print(confusion_matrix)


    for folder in os.listdir(path_image):
        j=labels.get(folder)
        k=0
        l=0
        m=0
        n=0
        o=0
        print(j)
        i = 0
        nb_elements[j] = len(os.listdir(path_image +folder))
        print(folder)
        for elements in os.listdir(path_image + folder):
            img = Image.open(path_image + folder + '/' + elements)
            img = load_image(img)/255
            y_prob = Model.predict(img)
            y_classes = y_prob.argmax(axis=-1)
            attributed[y_classes[0]] = attributed[y_classes[0]]+1
            if j == y_classes[0]:
                i+=1
            if 0==y_classes[0]:
                k+=1
            elif 1==y_classes[0]:
                l+=1
            elif 2==y_classes[0]:
                m+=1
            elif 3==y_classes[0]:
                n+=1
            elif 4==y_classes[0]:
                o+=1
        confusion_matrix[j,:]+=[k,l,m,n,o]
        correctly_attributed[j]=i

    print('nb_elements=', nb_elements )
    print('correctly_attributed=', correctly_attributed, "\n")
    print('confusion_matrix:\n',confusion_matrix, "\n")
    correctly_attributed = np.asarray(correctly_attributed)
    attributed = np.asarray(attributed)
    nb_elements = np.asarray(nb_elements)

    precision=np.asarray(correctly_attributed)/attributed
    recall=correctly_attributed/nb_elements
    F1=2*(precision*recall)/(precision+recall)

    print("Precision=", precision, '\n Recall=', recall, "\n F1=", F1)
    return

def F1_pretrained(path_image: object) -> object: # Version pour VGG16_newVersion
    nb_classes=len(os.listdir(path_image))
    correctly_attributed=nb_classes*[0]
    attributed=nb_classes*[0]
    nb_elements=nb_classes*[0]
    confusion_matrix= np.zeros((nb_classes,nb_classes))
    print(confusion_matrix)


    for folder in os.listdir(path_image):
        j=labels.get(folder)
        a = 0
        k=0
        l=0
        m=0
        n=0
        o=0
        print(j)
        i = 0
        nb_elements[j] = len(os.listdir(path_image +folder))
        print(folder)
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
            directory=r"/home/alexis51151/Documents/Bdd_multiclasse/Test",  # 159 elements
            target_size=(256, 256),
            color_mode="rgb",
            batch_size=1,  # Pour Geoffroy : 159
            class_mode="categorical",
        )
        print("Taille de test_generator" + str(len(test_generator)))
        for input, label in test_generator:
            print("Image n° " + str(a))
            a += 1
            features = np.zeros(shape=(1, 8, 8, 512))
            features = conv_base.predict(input)
            features = np.reshape(features, (1, 8 * 8 * 512))
            y_prob = model.predict(features)
            y_classes = y_prob.argmax(axis=-1)
            attributed[y_classes[0]] = attributed[y_classes[0]]+1
            if j == y_classes[0]:
                i+=1
            if 0==y_classes[0]:
                k+=1
            elif 1==y_classes[0]:
                l+=1
            elif 2==y_classes[0]:
                m+=1
            elif 3==y_classes[0]:
                n+=1
            elif 4==y_classes[0]:
                o+=1
            if (a==160):
                break

        confusion_matrix[j,:]+=[k,l,m,n,o]
        correctly_attributed[j]=i

    print('nb_elements=', nb_elements )
    print('correctly_attributed=', correctly_attributed, "\n")
    print('confusion_matrix:\n',confusion_matrix, "\n")
    correctly_attributed = np.asarray(correctly_attributed)
    attributed = np.asarray(attributed)
    nb_elements = np.asarray(nb_elements)

    precision=np.asarray(correctly_attributed)/attributed
    recall=correctly_attributed/nb_elements
    F1=2*(precision*recall)/(precision+recall)

    print("Precision=", precision, '\n Recall=', recall, "\n F1=", F1)
    return



# Simple function to load an image
def load_image(img):
    img = img.resize((64,64))
    img = np.array(img)
    img = np.reshape(img, (1,64,64,3))
    return img

F1_pretrained(path_image)