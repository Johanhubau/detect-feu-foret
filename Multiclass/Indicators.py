import os
import numpy as np
import keras
from PIL import Image

labels= {'Fire': 0, 'Fog': 2, 'Not_fire': 1, 'Red_object': 3, 'Smoke': 4} # Our neural network classes


# Here you load your trained network
path_model="/home/alexis51151/detect-feu-foret/Models/MulticlassV1"
model = keras.models.load_model(path_model)

# Here you load your image you want to visualize when passed through the neural network
path_image = "/home/alexis51151/Documents/Bdd_multiclasse/Test/"


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
        o=0
        print(j)
        i = 0
        nb_elements[j] = len(os.listdir(path_image +folder))
        print(folder)
        for elements in os.listdir(path_image + folder):
            img = Image.open(path_image + folder + '/' + elements)
            img = load_image(img)/255
            y_prob = model.predict(img)
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

# Simple function to load an image
def load_image(img):
    img = img.resize((75,75))
    img = np.array(img)
    img = np.reshape(img, (1,75,75,3))
    return img

F1(path_image)