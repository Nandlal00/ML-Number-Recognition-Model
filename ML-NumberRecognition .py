#!/usr/bin/env python
# coding: utf-8

# In[29]:


# importing the required libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# In[32]:


#  split Data into training and testing sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# In[4]:


# normalize the pixel values and reshape the input data.
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# In[5]:


# Create a convolutional neural network (CNN) model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


# In[6]:


# Compile the model by specifying the loss function, optimizer, and metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[7]:


# Now train the model on the training data
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))


# In[8]:


# Evaluate the model's performance
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc*100:.2f}%")


# In[9]:


# use the trained model to make predictions on new images
predictions = model.predict(x_test)


# In[10]:


# visualize the test images along with their predicted labels
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(f"Predicted: {np.argmax(predictions[i])}, Actual: {y_test[i]}")
plt.show()

