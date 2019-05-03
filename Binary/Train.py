from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten
import numpy as np
from matplotlib import pyplot as plt
import keras

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=r"/home/geoffroy/Documents/Gate/Biclasse/Train", #1528 elements,
    target_size=(64, 64),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    directory=r"/home/geoffroy/Documents/Gate/Biclasse/Test", #159 elements
    target_size=(64, 64),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=37
)

validate_datagen = ImageDataGenerator(rescale=1./255)

predict_generator = test_datagen.flow_from_directory(
    directory=r"/home/geoffroy/Documents/Gate/Internet",
    target_size=(64, 64),
    color_mode="rgb",
    batch_size=1,
    class_mode="categorical",
    shuffle=True,
    seed=35
)



# path_model="/home/geoffroy/Documents/Gate/bigger_filters_augmented"
# model = keras.models.load_model(path_model)


# hist_internet= model.predict_generator(
# predict_generator,
# steps=8
# )




model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


hist= model.fit_generator(
    train_generator,
    steps_per_epoch=int(1528//32),
    epochs=15,
    validation_data=test_generator,
    validation_steps=1)

plt.plot(hist.epoch, hist.history['val_loss'], label="val_loss")
plt.plot(hist.epoch, hist.history['loss'], label='loss')
plt.legend()
plt.show()

model.save('/home/geoffroy/Documents/Gate/Biclasse')


