from PIL import Image
import numpy as np
import keras
import os
import modelconfig as cfg

MODEL_SAVE_NAME = "Lolipop"
MODEL_LOAD_NAME = "Lolipop"

SAVE_PATH = cfg.paths['save'] + MODEL_SAVE_NAME
MODEL_PATH = cfg.paths['model'] + MODEL_LOAD_NAME


TRAIN_PATH = cfg.paths['traindb']
TEST_PATH = cfg.paths['testdb']
VAL_PATH = cfg.paths['validatedb']

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

    path_model = MODEL_PATH
    path_image = VAL_PATH
    model = keras.models.load_model(path_model)
    test_internet(path_image)