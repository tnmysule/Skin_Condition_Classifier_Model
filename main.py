import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
import random
from tensorflow.keras.models import load_model
from preprocessing import scaling, get_split, visualise_images
from model import Skin_model, training,eval, testing

print()
print()
print('data collection started')
data = tf.keras.utils.image_dataset_from_directory("Skin_Conditions")
print()
print()
print('data collection completed')

x_train, y_train, x_test, y_test, class_names = get_split(data)
x_train, x_test = scaling(x_train,x_test)
#visualise_images(x_train, y_train, class_names)

print()
print()
print('data splitting completed')
print()
print()

model_path = 'skin_model.h5'
initial_epoch = 100
epochs = 200

# Check if the model file exists
if os.path.exists(model_path):
  # Load the model
  model = load_model(model_path)
  model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(), optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), metrics = ['accuracy'])
  print("Model loaded successfully.")
else:
  # If the model does not exist, define and train a new model
  print("Model not found. Training a new model...")
  model = Skin_model()
  print()
  print()
  print('model created, starting training')
  print()
  print()

hist, model = training(model, x_train, y_train, x_test,y_test, initial_epoch, epochs)
print()
print()
print('training done')

#performance_loss(hist)
#performance_accuracy(hist)
#visualise_images(x_test, y_test, class_names)
eval(x_test,y_test, model, class_names)
#print(f'Precision:{pre.result().numpy()},Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')

#image = 'test_images/rosacea2.jpg'
#target_size = (256,256)


#testing(image, model, class_names, target_size)

model.save('skin_model.h5', include_optimizer=True)

