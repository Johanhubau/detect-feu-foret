from keras.preprocessing.image import ImageDataGenerator, image
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization,Conv2D, MaxPooling2D, Flatten
from keras import layers
import numpy as np
from matplotlib import pyplot as plt
import os
import keras
import keras.backend as K

batch_size=64
target_size=256

# Normalise et data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=5, fill_mode='reflect', horizontal_flip=True)
validate_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


#Affiche un échantillon d'image
fnames = [os.path.join('/home/geoffroy/Documents/Gate/Bdd_perso/Train/Train_smoke', fname) for fname in os.listdir('/home/geoffroy/Documents/Gate/Bdd_perso/Train/Train_smoke')]
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


predict_generator = test_datagen.flow_from_directory(
    directory=r"/home/geoffroy/Documents/Gate/Internet",
    target_size=(target_size, target_size),
    color_mode="rgb",
    batch_size=1,
    class_mode="categorical",
    shuffle=True,
    seed=35
)

print(train_generator.class_indices)

# path_model="/home/geoffroy/Documents/Gate/Modèles/model_biclasse"
# model = keras.models.load_model(path_model)


# hist_internet= model.predict_generator(
# predict_generator,
# steps=8
# )




model = Sequential()
model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(target_size, target_size, 3))) #avant tout à 16
#model.add(layers.BatchNormalization())
model.add(Conv2D(16, (3, 3), activation='relu'))
#model.add(layers.BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(layers.BatchNormalization())
model.add(Conv2D(16, (3, 3), activation='relu'))
#model.add(layers.BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

#model.add(layers.BatchNormalization())
model.add(Dense(256, activation='relu'))  #Avant tout à 100
#model.add(layers.BatchNormalization())
model.add(Dense(128, activation='relu'))
#model.add(layers.BatchNormalization())
model.add(Dense(5, activation='softmax'))

#model.load_weights('/home/geoffroy/Documents/Gate/corsica')
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy', #loss='custom_categorical_crossentropy',
              metrics=['accuracy'])


hist= model.fit_generator(
    train_generator,
    steps_per_epoch=int(1525//batch_size),
    epochs=10,
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

model.save('/home/geoffroy/Documents/Gate/Modèles/MulticlassV1')


