from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os

gen = ImageDataGenerator(rotation_range=40, width_shift_range=20, height_shift_range=20,horizontal_flip=True)

path_origin="/home/geoffroy/Documents/Gate/Bdd_perso/Test/Fire/"
path_destination="/home/geoffroy/Documents/Gate/Bdd_perso_augmented/Test/Fire/"
error=0

for elements in os.listdir(path_origin):
    image = np.expand_dims(Image.open(path_origin + elements), 0)

    try:
        aug_iter = gen.flow(image)
        aug_images = [next(aug_iter)[0] for i in range(3)]

        for i in range(len(aug_images)):

            img=Image.fromarray(aug_images[i].astype(np.uint8))
            img.save(path_destination+str(i)+str(elements))
    except ValueError:
        print("Value error on " + path_origin + elements)

    except OSError:
        print("Os error on " + path_origin + elements)
        error=1


    finally:
        img = Image.open(path_origin + elements)
        if(error!=1):
            img.save(path_destination+str(elements))
        error=0