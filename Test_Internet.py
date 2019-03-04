from PIL import Image
import numpy as np
import keras
import os

def load_image(img):
    img = np.array(img)
    img = Image.fromarray(img)
    img = img.resize((64,64))
    img = np.array(img)
    img = np.reshape(img, (1,64,64,3))
    return img

def test_internet(path_image):
    for elements in os.listdir(path_image):
        print(elements)
        img = Image.open(path_image + elements)
        img = load_image(img)/255
        print(model.predict(img))
    return

if __name__ == '__main__':

    path_model = '/home/geoffroy/Documents/Gate/corsica'
    path_image = '/home/geoffroy/Documents/Gate/Internet/'
    model = keras.models.load_model(path_model)
    test_internet(path_image)