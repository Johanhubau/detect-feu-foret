from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras import models
from keras import layers
from keras import optimizers
from matplotlib import pyplot as plt
import numpy as np

# Normalise et data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=5, fill_mode='reflect', horizontal_flip=True)
validate_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

#Main code for the neural network starts here

batch_size=64 #Pour Geoffroy : 64
target_size=256
labels= {'Fire': 0, 'Fog': 1, 'Not_fire': 2, 'Red_object': 3, 'Smoke' : 4}


train_generator = train_datagen.flow_from_directory(
    directory=r"/home/geoffroy/Documents/Gate/Bdd_perso/Train", #1525 elements,
    target_size=(target_size, target_size),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)


test_generator = test_datagen.flow_from_directory(
    directory=r"/home/geoffroy/Documents/Gate/Bdd_perso/Test", #159 elements
    target_size=(target_size, target_size),
    color_mode="rgb",
    batch_size=64, #Pour Geoffroy : 159
    class_mode="categorical",
    shuffle=True,
    seed=37
)


predict_generator = validate_datagen.flow_from_directory(
    directory=r"/home/geoffroy/Documents/Gate/Internet", #45 éléments
    target_size=(target_size, target_size),
    color_mode="rgb",
    batch_size=45, #Pour Geoffroy : 46
    class_mode="categorical",
    shuffle=True,
    seed=35
)

# create the base pre-trained model
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(256,256,3))
conv_base.summary()


def extract_features(generator, sample_count):
    i = 0
    features = np.zeros(shape=(sample_count, 8, 8, 512))
    labels = np.zeros(shape=(sample_count, 4))
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i+1)*batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i+=1
        if i*batch_size>=sample_count:
            break
    return features,labels

train_features, train_labels=extract_features(train_generator, 1525)
test_features, test_labels=extract_features(test_generator, 159)
validation_features, validation_labels=extract_features(predict_generator, 45)

train_features = np.reshape(train_features, (1525, 8 * 8 * 512))
test_features = np.reshape(test_features, (159, 8 * 8 * 512))
validation_features = np.reshape(validation_features, (45, 8 * 8 * 512))

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=8 * 8 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

hist = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=64,
                    validation_data=(test_features, test_labels))


# # add new layers
# x = conv_base.output
# x = Flatten()(x)
# x = Dense(100, activation="relu")(x)
# x = Dropout(0.5)(x)
# x = Dense(100, activation="relu")(x)
# predictions = Dense(5, activation='sigmoid')(x)  # We want 5 classes as output

# # this is the model we will train
# model = Model(inputs=conv_base.input, outputs=predictions)
#
# # We only train the 5 last layers
# for layer in model.layers[0:5]:
#    layer.trainable = False
# for layer in model.layers[5:len(model.layers)]:
#    layer.trainable = True
#
# # compile the model (should be done *after* setting layers to non-trainable)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# # train the model on the new data for a few epochs
# hist= model.fit_generator(
#     train_generator,
#     steps_per_epoch=int(1525//batch_size),
#     epochs=5,
#     validation_data=test_generator,
#     validation_steps=1,
#     verbose=1)

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

model.save("/home/geoffroy/PycharmProjects/detect-feu-foret/Models/pretrained_VGG16_5classes")



