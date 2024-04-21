#!/usr/bin/env python3
#
# Compute Sobel gradient to images (tensorflow)

import numpy as np
import matplotlib.pyplot as plt

# We use tensorflow: install this with `pip3 install tensorflow` or similar
import tensorflow as tf

# Load example dataset (actually for machine learning, but works for this, too)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# We only want 16 random example images
images = x_train[np.random.randint(x_train.shape[0], size=16),...] # 16x32x32x3 tensor
# Turn into greyscale
images = tf.image.rgb_to_grayscale(images) # 16x32x32x1 tensor
# Normalise to 0..1
images = tf.cast(images, tf.float32) / 255.0

# Sobel gradient:
# Convolution with Sobel filter
#     [[-1,0,1]
#      [-2,0,2]
#      [-1,0,1]]
# in x and y direction.
# Then take norm of x/y value per pixel.

# Sobel filter (in x; y is transposed)
sobel_x = tf.constant([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]],
                       tf.float32)


# Turn from 3x3 into 1x1x3x3 (height,width,inp_chs,out_chs) filter
sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
# Sobel y filter
sobel_y = tf.transpose(sobel_x, [1, 0, 2, 3])

# Apply convolution for x and y direction
# 移动距离
g_x = tf.nn.conv2d(images, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
g_y = tf.nn.conv2d(images, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
# Compute norm
g = tf.math.sqrt(tf.math.pow(g_x,2)+tf.math.pow(g_y,2))






# Show images and results
fig, ax = plt.subplots(4, 1, figsize=(8,8))
def imshow(imgs,ax,cmap):
  # Create a list of 16 entries of shape 32x32x1
  t = tf.unstack(imgs[:8 * 2], num=16, axis=0)
  # Concatenate 8 elements of the list to a row for two rows
  rows = [tf.concat(t[l*8:(l+1)*8], axis=1) for l in range(0,2)]
  # Concatenate the two rows
  image = tf.concat(rows, axis=0)
  ax.imshow(image, cmap=cmap, vmin=0, vmax=1)
imshow(images,ax[0],cmap="gray")
imshow(g_x,ax[1],cmap="viridis")
imshow(g_y,ax[2],cmap="viridis")
imshow(g,ax[3],cmap="viridis")
plt.show()
