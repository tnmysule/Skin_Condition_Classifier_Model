import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing import image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

def Skin_model():
  model = Sequential()

  # Input layer and first convolutional block
  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
  model.add(MaxPooling2D((2, 2)))
  model.add(BatchNormalization())

  # Second convolutional block
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D((2, 2)))
  model.add(BatchNormalization())

  # Third convolutional block
  model.add(Conv2D(128, (3, 3), activation='relu'))
  model.add(MaxPooling2D((2, 2)))
  model.add(BatchNormalization())

  # Flatten and fully connected layers
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.5))  # Dropout to reduce overfitting

  model.add(Dense(64, activation='relu'))
  model.add(Dropout(0.5))

  # Output layer with softmax for multiclass classification
  model.add(Dense(6, activation='softmax'))  # Assume 5 classes

  model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(), optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), metrics = ['accuracy'])
  print()
  print()
  model.summary()
  print()
  print()
  return model

def training(model, x_train, y_train, x_test,y_test,initial_epoch, epochs):
  logdir = 'logs'
  tensorflow_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir)
  hist = model.fit(x_train, y_train, epochs = epochs, initial_epoch = initial_epoch, validation_data = (x_test, y_test), callbacks = [tensorflow_callback])

  return hist, model

def eval(x_test,y_test, model, class_names):
  y_pred = model.predict(x_test)
  print(y_pred)
  print('printing predictions of first 10 images')
  
  for i in range(10):
    print()
    print('prediction array for image ',i)
    print(y_pred[i])
    print()
    print(f'image of class {y_test[i]} was predicted as image of class {y_pred[i].argmax()}')
  cm = confusion_matrix(y_test, y_pred.argmax(axis=1))
  display = ConfusionMatrixDisplay(cm, display_labels = class_names)

  fig, ax = plt.subplots(figsize = (10,10))
  display.plot(ax = ax)
  plt.show()

def testing(image_path, model, class_names, target_size):
  img = image.load_img(image_path, target_size=target_size)
    
  # Convert the image to a numpy array
  img_array = image.img_to_array(img)
  
  # Expand dimensions to match the model's input shape
  img_array = np.expand_dims(img_array, axis=0)
  
  # Normalize the image (if your model was trained on normalized images)
  img_array /= 255.0

  pred = model.predict(img_array)
  val = np.argmax(pred, axis = 1)[0]
  percent = round(pred[0].max()*100, 2)
  plt.imshow(img)  # Display the image (converting to unsigned 8-bit integer if necessary)
  plt.title(f'Predicted: {class_names[val]}, Probability: {percent}%')
  plt.axis('off')  # Hide the axes
  plt.show()

  print(f'The image is of class {class_names[val]}')
