import os
import numpy as np
import keras
from PIL import Image
from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.layers import Flatten

labels= {'Fire': 0, 'Fog': 1, 'Not_fire': 2, 'Red_object': 3} # Our neural network classes


# Here you load your trained network
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(256,256,3)) #Si modèle avec base VGG1, on charge cette base
path_model="/home/geoffroy/PycharmProjects/detect-feu-foret/Models/pretrained_VGG16_5classes"
model = keras.models.load_model(path_model)

# Here you load your image you want to visualize when passed through the neural network
path_image = "/home/geoffroy/Documents/Gate/Bdd_perso/Test/"


# And then you launch this function to get the indicators
def F1(path_image):

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
        print(j)
        i = 0
        nb_elements[j] = len(os.listdir(path_image +folder))
        print(folder)
        for elements in os.listdir(path_image + folder):
            img = Image.open(path_image + folder + '/' + elements)
            img = load_image(img)/255

            y_feature = np.zeros(shape=(1, 8, 8, 512))
            y_feature[0]=conv_base.predict(img) #Pour modèle avec base VGG16 on extrait les features
            y_feature = np.reshape(y_feature, (1, 8 * 8 * 512))
            y_prob = model.predict(y_feature) #y_feature si base vgg16, img sinon
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
            #elif 4==y_classes[0]:
             #   o+=1

        confusion_matrix[j,:]+=[k,l,m,n]
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
    img = img.resize((256,256))
    img = np.array(img)
    img = np.reshape(img, (1,256,256,3))
    return img


F1(path_image)