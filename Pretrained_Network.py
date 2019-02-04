from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten
from matplotlib import pyplot as plt
import modelconfig as cfg



## Chargement des datasets

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    directory=cfg.paths["traindb"], #2443 elements
    target_size=(64, 64),
    color_mode="rgb",
    batch_size=50,
    class_mode="binary",
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    directory=cfg.paths["testdb"], #320 elements
    target_size=(64, 64),
    color_mode="rgb",
    batch_size=50,
    class_mode="binary",
    shuffle=True,
    seed=37
)

validate_datagen = ImageDataGenerator(rescale=1./255)
predict_generator = test_datagen.flow_from_directory(
    directory=cfg.paths["validatedb"],
    target_size=(64, 64),
    color_mode="rgb",
    batch_size=1,
    class_mode="binary",
    shuffle=True,
    seed=35
)


# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
# x = GlobalAveragePooling2D()(x)
predictions = Dense(1, activation='sigmoid')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers[0:len(model.layers)-3]:
   layer.trainable = False
for layer in model.layers[len(model.layers)-3:len(model.layers)]:
   layer.trainable = True

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])
# train the model on the new data for a few epochs
hist = model.fit_generator(train_generator,
steps_per_epoch=50,
epochs=10,
validation_data=test_generator,
validation_steps=1)

plt.plot(hist.epoch, hist.history['val_loss'], label="val_loss")
plt.plot(hist.epoch, hist.history['loss'], label='loss')
plt.legend()
plt.show()

model.save(cfg.paths["save"])



