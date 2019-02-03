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
    i=0
    prediction = 0
    for elements in os.listdir(path_image):
        i=i+1
        print(elements)
        img = Image.open(path_image + elements)
        img = load_image(img)/255
        prediction +=  model.predict(img)[0][0]
        print(prediction/i)
    return

if __name__ == '__main__':

    path_model = '/home/geoffroy/Documents/Gate/big_model_30epoch_plusdefiltres_augmented'
    path_image = '/home/geoffroy/Documents/Gate/Internet/'
    model = keras.models.load_model(path_model)
    test_internet(path_image)