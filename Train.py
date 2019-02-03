from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten
from matplotlib import pyplot as plt
import keras
import modelconfig as cfg

MODEL_SAVE_NAME = "Lolipop"
MODEL_LOAD_NAME = "Lolipop"

SAVE_PATH = cfg.paths['save'] + MODEL_SAVE_NAME
MODEL_PATH = cfg.paths['model'] + MODEL_LOAD_NAME


TRAIN_PATH = cfg.paths['traindb']
TEST_PATH = cfg.paths['testdb']
VAL_PATH = cfg.paths['validatedb']


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=TRAIN_PATH, #2443 elements
    target_size=(64, 64),
    color_mode="rgb",
    batch_size=50,
    class_mode="binary",
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    directory=TEST_PATH, #320 elements
    target_size=(64, 64),
    color_mode="rgb",
    batch_size=50,
    class_mode="binary",
    shuffle=True,
    seed=37
)

validate_datagen = ImageDataGenerator(rescale=1./255)
predict_generator = test_datagen.flow_from_directory(
    directory=VAL_PATH,
    target_size=(64, 64),
    color_mode="rgb",
    batch_size=1,
    class_mode="binary",
    shuffle=True,
    seed=35
)

# path_model= MODEL_PATH
# model = keras.models.load_model(path_model)
#
#
# hist_internet= model.predict_generator(
# predict_generator,
# steps=8
# )
#

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(64, 64, 3)))
model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(BatchNormalization())


model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])



hist= model.fit_generator(
train_generator,
steps_per_epoch=50,
epochs=3,
validation_data=test_generator,
validation_steps=1)

plt.plot(hist.epoch, hist.history['val_loss'], label="val_loss")
plt.plot(hist.epoch, hist.history['loss'], label='loss')
plt.legend()
plt.show()

model.save(SAVE_PATH)


