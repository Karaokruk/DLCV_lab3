from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import os


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

num_class = 10

y_train = keras.utils.to_categorical(y_train,num_class)
y_test = keras.utils.to_categorical(y_test,num_class)

model = Sequential()

model.add(Conv2D(32, kernel_size=4, activation='elu', input_shape=(32,32,3),padding='same'))
model.add(Conv2D(32, kernel_size=4, activation='elu', padding='same'))
model.add(MaxPooling2D((2,2), padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=4, activation='elu', input_shape=(32,32,3),padding='same'))
model.add(Conv2D(64, kernel_size=4, activation='elu', padding='same'))
model.add(MaxPooling2D((2,2), padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(128, kernel_size=4, activation='elu', input_shape=(32,32,3),padding='same'))
model.add(Conv2D(128, kernel_size=4, activation='elu',padding='same'))
model.add(MaxPooling2D((2,2), padding='same'))
model.add(Dropout(0.5))


model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(x_train, y_train, validation_split = 0.2,epochs=100 , batch_size=150)


score = model.evaluate(x_test, y_test, batch_size=128)
print("score",score)
print("Neural network accuracy: %.2f%%" % (score[1]*100))

model.predict(x_test[:4])

print(y_test[:4])

plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
