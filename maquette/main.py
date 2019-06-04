from flask import Flask, render_template
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import random as rdm
from shutil import copyfile
import glob
import time
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(256,256,3))

MAIN_PATH = "/home/heisenberg/gate/DB/Bdd_perso/Test/"

class_names = ["Feu", "Fumée ou brouillard", "Non feu", "Objet rouge"]
class_folder_names = ["Fire/", "Fog/", "Not_fire/", "Red_object/"]

app = Flask(__name__)

def read_classes():
    file = open("net.txt", "r")
    array = []
    lines = file.readlines()
    print(lines)
    for line in lines:
        array.append(line[:-1])
    chosen_class = array.pop()
    file.close()
    return array, chosen_class

#l'envoyer à la page et dans le réseau de neuronne(pour l'instant fausse fonction dico)
def resultat() :
    tableau_res = {}
    tableau_res['feu']= rdm.uniform(0,1)
    tableau_res['pas feu'] = rdm.uniform(0,1)
    tableau_res['objet rouge']=rdm.uniform(0,1)
    tableau_res['brouillard']=rdm.uniform(0,1)
    return tableau_res

def erase_txt():
    fichier = open("backup.txt", "w")
    fichier.close()
    return

def get_array():
    file = open("backup.txt", "r")
    array = []
    lines = file.readlines()
    if lines == []:
        return [[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]]
    for line in lines:
        line = line[:-1]
        print(line)
        subarray = line.split(',')
        print(subarray)
        for i in range(len(subarray)):
            subarray[i] = int(subarray[i])
        array.append(subarray)
    file.close()
    return array

def save_array(array):
    file = open("backup.txt", "w")
    for line in array:
        file.write(str(line[0]) + "," + str(line[1]) + "," + str(line[2]) + "," + str(line[3]) + "\n")
    file.close()

def make_confusion_matrix(array):
    array = np.array(array)
    array = (array.T/np.sum(array, axis=1)).T
    df_cm = pd.DataFrame(array, ["Fire","Smoke/Fog","Not Fire","Red object"],
                      ["Fire","Smoke/Fog","Not Fire","Red object"])
    plt.figure(figsize = (10,8))
    sn.set(font_scale=1.4)#for label size
    ax = sn.heatmap(df_cm, annot=True,annot_kws={"size": 16,})# font size
    plt.xlabel('Classe prédite')
    plt.ylabel('Vraie classe')
    plt.savefig("/home/heisenberg/gate/detect-feu-foret/maquette/static/matrix.jpg")


@app.route("/")
def main():

    classes, chosen_class = read_classes()

    max = 0
    detected = -1
    #Get detected class
    print(classes)
    for i in range(len(classes)):
        if float(classes[i]) > max:
            detected = i
            max = float(classes[1])
    assert (detected != -1), "Something's wrong with the predictions"

    #Get array, modify it, make confusion matrix and save array
    array = get_array()
    array[int(chosen_class)][detected] += 1
    make_confusion_matrix(array)
    save_array(array)

    #Sanitize predictions for display
    predictions = []
    for i in range(len(classes)):
        pred = str(float(classes[i])*100) + '%'
        predictions.append((class_names[i], pred))

    return render_template('index.html', predictions=predictions, expected_result=class_names[int(chosen_class)])


if __name__ == "__main__":
   app.run(debug = True)
