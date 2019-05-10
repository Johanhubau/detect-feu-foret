from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten
from matplotlib import pyplot as plt


# Normalise et data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=5, fill_mode='reflect', horizontal_flip=True)
validate_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

#Main code for the neural network starts here

batch_size=32 #Pour Geoffroy : 64
target_size=256
labels= {'Fire': 0, 'Fog': 1, 'Not_fire': 2, 'Red_object': 3, 'Smoke' : 4}

train_generator = train_datagen.flow_from_directory(
    directory=r"/home/alexis51151/Documents/Bdd_multiclasse/Train", #1525 elements,
    target_size=(target_size, target_size),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)


test_generator = test_datagen.flow_from_directory(
    directory=r"/home/alexis51151/Documents/Bdd_multiclasse/Test", #159 elements
    target_size=(target_size, target_size),
    color_mode="rgb",
    batch_size=40, #Pour Geoffroy : 159
    class_mode="categorical",
    shuffle=True,
    seed=37
)


predict_generator = validate_datagen.flow_from_directory(
    directory=r"/home/alexis51151/detect-feu-foret/Internet",
    target_size=(target_size, target_size),
    color_mode="rgb",
    batch_size=20, #Pour Geoffroy : 46
    class_mode="categorical",
    shuffle=True,
    seed=35
)

# create the base pre-trained model
base_model = VGG16(weights='imagenet', include_top=False)
base_model.summary()
# add new layers
x = base_model.output
x = Flatten()(x)
x = Dense(100, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(100, activation="relu")(x)
predictions = Dense(5, activation='sigmoid')(x)  # We want 5 classes as output

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# We only train the 5 last layers
for layer in model.layers[0:5]:
   layer.trainable = False
for layer in model.layers[5:len(model.layers)]:
   layer.trainable = True

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# train the model on the new data for a few epochs
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
plt.savefig('loss')

plt.figure()
plt.plot(hist.epoch, hist.history['val_acc'], 'm', label="val_acc")
plt.plot(hist.epoch, hist.history['acc'], 'b', label='acc')
plt.title('Accuracy')
plt.legend()
plt.savefig('acc')

plt.show()

model.save("/home/alexis51151/detect-feu-foret/Models/pretrained_VGG16_5classes")



