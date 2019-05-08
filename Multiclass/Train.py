from keras.preprocessing.image import ImageDataGenerator, image
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization,Conv2D, MaxPooling2D, Flatten
from keras import layers
import numpy as np
from matplotlib import pyplot as plt
import os
import keras
import keras.backend as K
from PIL import Image
from itertools import product
from functools import partial

def load_image(img):
    img = img.resize((256,256))
    img = np.array(img)
    img = np.reshape(img, (1,256,256,3))
    return img


#Main code for the neural network starts here


batch_size=64
target_size=256
labels= {'Fire': 0, 'Fog': 1, 'Not_fire': 2, 'Red_object': 3}


# Normalise et data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=5, fill_mode='reflect', horizontal_flip=True)
validate_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

"""
#Affiche un échantillon d'image
fnames = [os.path.join('/home/geoffroy/Documents/Gate/Bdd_perso/Train/Fog', fname) for fname in os.listdir('/home/geoffroy/Documents/Gate/Bdd_perso/Train/Fog')]
img_path = fnames[2]
img = image.load_img(img_path, target_size=(target_size,target_size))
x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)
i=0
for batch in train_datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i+=1
    if i % 4 == 0:
        break

plt.show()
"""
train_generator = train_datagen.flow_from_directory(
    directory=r"/home/geoffroy/Documents/Gate/Bdd_perso/Train", #1525 elements,
    target_size=(target_size, target_size),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

#labels = dict((v,k) for k,v in labels.items())
#predictions = [labels[k] for k in predicted_class_indices]

test_generator = test_datagen.flow_from_directory(
    directory=r"/home/geoffroy/Documents/Gate/Bdd_perso/Test", #159 elements
    target_size=(target_size, target_size),
    color_mode="rgb",
    batch_size=159,
    class_mode="categorical",
    shuffle=True,
    seed=37
)


predict_generator = validate_datagen.flow_from_directory(
    directory=r"/home/geoffroy/Documents/Gate/Internet",
    target_size=(target_size, target_size),
    color_mode="rgb",
    batch_size=46,
    class_mode="categorical",
    shuffle=True,
    seed=35
)

print(train_generator.class_indices)

# path_model="/home/geoffroy/Documents/Gate/Modèles/MulticlassV3"
# model = keras.models.load_model(path_model)


# hist_internet = model.predict_generator(
# predict_generator,
# steps=8
# )




model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(target_size, target_size, 3))) #avant tout à 16
#model.add(layers.BatchNormalization())
model.add(Conv2D(16, (3, 3), activation='relu'))
#model.add(layers.BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(layers.BatchNormalization())
model.add(Conv2D(16, (3, 3), activation='relu'))
#model.add(layers.BatchNormalization())
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

#model.add(layers.BatchNormalization())
model.add(Dense(100, activation='relu'))  #Avant tout à 100
#model.add(layers.BatchNormalization())
model.add(Dense(100, activation='relu'))
#model.add(layers.BatchNormalization())
model.add(Dense(4, activation='softmax'))

#model.load_weights('/home/geoffroy/Documents/Gate/corsica')
model.summary()

model.compile(optimizer='adam',#loss=ncce,
               loss='categorical_crossentropy',
              metrics=['accuracy'])


hist= model.fit_generator(
    train_generator,
    steps_per_epoch=int(1525//batch_size),
    epochs=5,
    validation_data=test_generator,
    validation_steps=1,
    verbose=1)

plt.plot(hist.epoch, hist.history['val_loss'], 'm', label="val_loss")
plt.plot(hist.epoch, hist.history['loss'], 'b', label='loss')
plt.title('Loss')
plt.legend()
plt.savefig('custom_loss')

plt.figure()
plt.plot(hist.epoch, hist.history['val_acc'], 'm', label="val_acc")
plt.plot(hist.epoch, hist.history['acc'], 'b', label='acc')
plt.title('Accuracy')
plt.legend()
plt.savefig('custom_acc')

plt.show()

model.save('/home/geoffroy/Documents/Gate/Modèles/MulticlassV4')

F1("/home/geoffroy/Documents/Gate/Bdd_perso/Test/")
