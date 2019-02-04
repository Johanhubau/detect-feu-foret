from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import load_model
from matplotlib import pyplot as plt
# import modelconfig as cfg # Pour utiliser notre propre Dataset
import modelconfigALG_CorsicaDB as cfg# Pour utiliser la Corsica Fire database

# Charger notre réseau

#model = load_model(cfg.paths["save"]+"\Lolipop1")

## Chargement des datasets

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    directory=cfg.paths["traindb"], #2443 elements
    target_size=(75, 75),
    color_mode="rgb",
    batch_size=50,
    class_mode="binary",
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    directory=cfg.paths["testdb"], #320 elements
    target_size=(75, 75),
    color_mode="rgb",
    batch_size=50,
    class_mode="binary",
    shuffle=True,
    seed=37
)

validate_datagen = ImageDataGenerator(rescale=1./255)
predict_generator = test_datagen.flow_from_directory(
    directory=cfg.paths["validatedb"],
    target_size=(75, 75),
    color_mode="rgb",
    batch_size=1,
    class_mode="binary",
    shuffle=True,
    seed=35
)


# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(75, 75, 3))

# add a global spatial average pooling layer
x = base_model.output
x = Flatten()(x)
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
steps_per_epoch=22,
epochs=10,
validation_data=test_generator,
validation_steps=5)

plt.plot(hist.epoch, hist.history['val_loss'], label="val_loss")
plt.plot(hist.epoch, hist.history['loss'], label='loss')
plt.legend()
plt.show()

model.save(cfg.paths["save"])



