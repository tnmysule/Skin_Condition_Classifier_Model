import numpy as np
import random
from matplotlib import pyplot as plt

def get_split(data):
  class_names = data.class_names

  print('classes: ',class_names)

# Initialize lists to store the images and labels
  xtemp_train = []
  ytemp_train = []

  # Iterate over the dataset
  for images, labels in data:
      xtemp_train.append(images.numpy())  # Convert tensor to numpy array
      ytemp_train.append(labels.numpy())  # Convert tensor to numpy array

  # Convert lists to numpy arrays
  xtemp_train = np.concatenate(xtemp_train, axis=0)
  ytemp_train = np.concatenate(ytemp_train, axis=0)

  # Optionally, ensure labels are within the range of 0 to 5
  ytemp_train = np.clip(ytemp_train, 0, 5)

  print(f"image set shape: {xtemp_train.shape}")
  print(f"label set shape: {ytemp_train.shape}")

  print('data length: ', len(xtemp_train))
  train_size = int(len(xtemp_train)*0.7)

  x_train = xtemp_train[0:train_size]
  x_test = xtemp_train[train_size:len(xtemp_train)]

  y_train = ytemp_train[0:train_size]
  y_test = ytemp_train[train_size:len(xtemp_train)]
  print(f'training images: {len(x_train)}, training labels: {len(y_train)}, testing images: {len(x_test)}, testing labels: {len(y_test)}')

  return x_train, y_train, x_test, y_test, class_names

def visualise_images(x_train, y_train, class_names):
    plt.figure(figsize=(6, 6))

    unique_classes = np.unique(y_train)  # Get unique class labels

    for i, class_label in enumerate(unique_classes):
        ax = plt.subplot(3, 2, i + 1)
        
        # Find the indices of all images of the current class
        class_indices = np.where(y_train == class_label)[0]
        
        # Randomly select one image from the current class
        image_index = random.choice(class_indices)
        
        plt.imshow(x_train[image_index], cmap=plt.cm.binary)
        plt.title(class_names[class_label])
        plt.axis('off')  # Optionally hide the axes

    plt.tight_layout()  # Adjust subplots to fit in the figure area.
    plt.show()  # Display all images at once


def scaling(x_train, x_test):
  x_train = x_train / 255
  x_test = x_test / 255
  return x_train,x_test