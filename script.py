import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()

print(digits.target[100])

# Figure size (width, height)

fig = plt.figure(figsize=(6, 6))

# Adjust the subplots 

fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 images

for i in range(64):

    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position

    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])

    # Display an image at the i-th position

    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

    # Label the image with the target value

    ax.text(0, 7, str(digits.target[i]))

plt.show()


model = KMeans(n_clusters=10)
model.fit(digits.data)

fig = plt.figure(figsize=(8, 3))

fig.suptitle('Cluster Center Images', fontsize = 14, fontweight = 'bold')

for i in range(10):
  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)

  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

plt.show()  

new_samples = np.array([
[0.00,1.60,6.86,7.62,6.71,2.75,0.00,0.00,0.00,6.03,6.94,3.43,6.33,7.63,1.76,0.00,0.00,5.64,7.40,4.81,4.19,7.63,4.50,0.00,0.00,0.69,4.80,6.62,7.09,7.62,4.19,0.00,0.00,0.00,0.00,0.00,4.12,7.62,1.60,0.00,0.00,0.00,0.00,1.14,7.47,5.19,1.60,5.80,0.00,3.13,6.71,7.62,7.62,7.17,6.79,7.32,1.07,7.55,7.62,7.02,3.20,4.73,6.33,4.57],
[0.00,0.00,0.00,0.69,2.67,3.05,1.68,0.00,0.00,0.00,2.21,7.40,7.62,7.62,7.47,3.20,0.00,0.31,6.86,6.41,1.45,1.37,7.32,5.19,0.00,1.60,7.62,2.67,0.00,0.99,7.63,3.81,0.00,2.90,7.62,1.22,0.00,4.19,7.63,1.14,0.00,3.05,7.62,0.76,0.08,6.64,6.10,0.00,0.00,2.90,7.62,4.80,5.11,7.62,3.05,0.00,0.00,0.23,4.81,6.86,6.79,4.43,0.08,0.00],
[0.00,0.08,2.14,3.05,2.21,0.00,0.00,0.00,0.00,4.50,7.62,7.62,7.63,2.44,0.00,0.00,0.00,6.10,7.62,7.24,7.62,3.05,0.00,0.00,0.00,2.29,3.81,4.57,7.62,2.36,0.00,0.00,0.00,0.00,0.00,2.14,7.62,1.90,1.15,0.92,0.00,0.00,0.00,3.97,7.49,0.61,6.56,5.26,0.00,4.57,7.63,7.62,6.48,3.05,7.62,2.90,0.00,6.86,7.32,6.10,7.62,7.62,6.10,0.15],
[0.00,0.00,0.00,0.38,4.04,4.57,2.67,0.00,0.00,0.00,0.08,5.64,7.55,6.71,7.62,2.59,0.00,0.00,1.91,7.62,3.35,0.84,7.62,3.05,0.00,0.00,3.51,7.62,0.84,1.15,7.62,3.05,0.00,0.00,3.81,7.62,0.00,3.51,7.62,1.52,0.00,0.00,3.51,7.63,1.14,6.33,6.63,0.00,0.00,0.00,1.45,7.55,7.62,7.62,2.67,0.00,0.00,0.00,0.00,1.60,3.05,2.14,0.00,0.00]
])

new_labels = model.predict(new_samples)
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')
