from PIL import Image
import numpy as np
import keras
import os

def load_image(img):
    img = img.resize((256,256))
    img = np.array(img)
    img = np.reshape(img, (1,256,256,3))
    return img

def test_internet(path_image):
    for elements in os.listdir(path_image):
        print(elements)
        img = Image.open(path_image + elements)
        img = load_image(img)/255
        y_prob = model.predict(img)
        y_classes = y_prob.argmax(axis=-1)
        print(y_prob)
        for key, value in labels.items():
            if value == y_classes[0]:
                print(key + '\n')
    return


if __name__ == '__main__':

    labels={'Fire': 0, 'Not_fire': 1, 'Train_fog': 2, 'Train_red': 3, 'Train_smoke': 4}

    path_model = '/home/geoffroy/Documents/Gate/Modèles/MulticlassV1'
    path_image = '/home/geoffroy/Documents/Gate/Internet/'
    model = keras.models.load_model(path_model)
    test_internet(path_image)