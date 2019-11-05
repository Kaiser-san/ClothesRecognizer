import keras
from keras.datasets import fashion_mnist
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Ucitavanje FashionMNIST skupa podataka
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Koristimo samo deo trening skupa (prvi od 10 fold-ova) radi efikasnosti treninga
skf = StratifiedKFold(n_splits=6, random_state=0, shuffle=False)
for train_index, test_index in skf.split(x_train, y_train):
    x_train, y_train = x_train[test_index], y_train[test_index]
    break
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#################################################################################
# U ovoj sekciji implementirati Keras neuralnu mrezu koja postize tacnost barem
# 85% na test skupu. Ne menjati fajl van ove sekcije.

x_train = (255-x_train)
x_test = (255-x_test)

x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard
from time import time

num_classes = 10
batch_size = 32
epochs = 25

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# Model

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Normalizacija

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(x_train[0].shape)

# Fitovanje
model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=[tensorboard])

# Evaluacija

scores = model.evaluate(x_test, y_test)
print('Test accuracy:', scores[1])

#################################################################################

# Cuvanje istreniranog modela u fajl
model.save('model.h5')