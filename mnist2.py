import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import cv2

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the dataset
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Reshape the data for the Convolutional Neural Network (CNN)
IMG_SIZE = 28
x_train_reshaped = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_test_reshaped = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print("Training Samples dimension", x_train_reshaped.shape)
print("Testing Samples dimension", x_test_reshaped.shape)

# Build the CNN model
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=x_train_reshaped.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dense(32))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("softmax"))

model.summary()

# Compile and train the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(x_train_reshaped, y_train, epochs=5, validation_split=0.3)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test_reshaped, y_test)
print("Test Loss on 10,000 test samples", test_loss)
print("Test Accuracy on 10,000 test samples", test_acc)



# Save the trained model
model.save('trained-model.h5')
