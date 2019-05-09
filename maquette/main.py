from flask import Flask, render_template
import random as rdm
import matplotlib.image as mpimg
import numpy as np

app = Flask(__name__)
#recup l'image
def recupimage(path) :
    img = mpimg.imread(path)
    if img.dtype == np.float32: # Si le résultat n'est pas un tableau d'entiers
        img = (img * 255).astype(np.uint8)
    return img

#l'envoyer à la page et dans le réseau de neuronne(pour l'instant fausse fonction dico)
def resultat() :
    tableau_res = {}
    tableau_res['feu']= rdm.uniform(0,1)
    tableau_res['pas feu'] = rdm.uniform(0,1)
    tableau_res['objet rouge']=rdm.uniform(0,1)
    tableau_res['brouillard']=rdm.uniform(0,1)
    return tableau_res

#matrice de confusion sur un fichier
def stockage(dic) :
    maxi=0
    for mot in dic :
        if dic[mot]> maxi :
            maxi=dic[mot]
            maxm=mot
    return maxm

def ecriture(mot) :
    fichier = open("backup.txt", "a")
    fichier.write(mot+" , ")
    fichier.close()
    return 

def erase_txt():
    fichier = open("backup.txt", "w")
    fichier.close()
    return

#zoe de test    
erase_txt()    
tabl=resultat()
print(tabl)
print(stockage(tabl))
ecriture(stockage(tabl))
tabl=resultat()
ecriture(stockage(tabl))
#fin zone de test
    



#
#@app.route("/")
#def main():
#
#    return render_template('index.html')
#
#
#if __name__ == "__main__":
#    app.run(debug = True)
#    
    
    

