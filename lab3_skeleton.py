from __future__ import print_function

#import tensorflow as tf
#import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Input
import numpy as np

#print('tensorflow:', tf.__version__)
#print('keras:', tensorflow.keras.__version__)


#load (first download if necessary) the MNIST dataset
# (the dataset is stored in your home direcoty in ~/.keras/datasets/mnist.npz
#  and will take  ~11MB)
# data is already split in train and test datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train : 60000 images of size 28x28, i.e., x_train.shape = (60000, 28, 28)
# y_train : 60000 labels (from 0 to 9)
# x_test  : 10000 images of size 28x28, i.e., x_test.shape = (10000, 28, 28)
# x_test  : 10000 labels
# all datasets are of type uint8


#To input our values in our network Dense layer, we need to flatten the datasets, i.e.,
# pass from (60000, 28, 28) to (60000, 784)
#flatten images
num_pixels = x_train.shape[1] * x_train.shape[2]
print("Num pixels : ", num_pixels)
x_train = x_train.reshape(x_train.shape[0], num_pixels)
x_test = x_test.reshape(x_test.shape[0], num_pixels)

#Convert to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalize inputs from [0; 255] to [0; 1]
x_train = x_train / 255
x_test = x_test / 255


#We want to have a binary classification: digit 0 is classified 1 and
#all the other digits are classified 0

y_new = np.zeros(y_train.shape)
y_new[np.where(y_train==0.0)[0]] = 1
y_train = y_new

y_new = np.zeros(y_test.shape)
y_new[np.where(y_test==0.0)[0]] = 1
y_test = y_new

num_classes = 1

def simpleNN(nb_epochs, batch_size):
    model = Sequential()
    model.add(Dense(1, input_dim=num_pixels, activation="sigmoid", kernel_initializer="normal"))
    model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
    model.summary()
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=nb_epochs, batch_size=batch_size)
    accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy : ", accuracy[1])

def hidden64NN(nb_epochs, batch_size): # TODO
    model = Sequential()
    #model.add(Input(shape=(None,num_pixels)))
    model.add(Dense(1, input_dim=num_pixels, activation="sigmoid", kernel_initializer="normal", name="simple_layer"))
    #model.add(Dense(64, input_dim=num_pixels, activation="sigmoid", kernel_initializer="normal", name="hidden_64sized_layer"))
    model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
    model.summary()
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=nb_epochs, batch_size=batch_size)
    accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy : ", accuracy[1])

def laDerniereQuestionLol():
    model = Model() # faut surement oublier Sequential pcq c'est lineaire (1 entree 1 sortie)

#Let start our work: creating a neural network
#First, we just use a single neuron.

nb_epochs = 5
batch_size = 128

#simpleNN(nb_epochs, batch_size)

hidden64NN(nb_epochs, batch_size)
