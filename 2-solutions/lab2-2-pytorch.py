#!/usr/bin/env python3
#
# Gaussian blur follow by sharpening filter with pytorch

import numpy as np
import matplotlib.pyplot as plt

# We use torch for this example and use example images from torchvision
# Install these packages with `pip3 install torch torchvision` or similar
import torch as t
import torchvision as tv

# Load example dataset (actually for machine learning, but works for this, too)
transf = tv.transforms.Compose([tv.transforms.Grayscale(), tv.transforms.ToTensor()])
imgset = tv.datasets.CIFAR10(root='./cifar10-data', train=True,
                             download=True, transform=transf)
imgloader = t.utils.data.DataLoader(imgset, batch_size=16,
                                    shuffle=True, num_workers=2)
# Get 16 (given by batch_size above) random (as we shuffle) images
dataiter = iter(imgloader)
images, _ = next(dataiter) # Two outputs, as this is an input/output pair for machine learning





# Gaussian Blur
# Smoothing Filter need权重矩阵中的元素通常表示像素的权重，
# 以平均周围像素的值。为了确保滤波器的响应不会导致图像过度模糊，
# 应该将权重矩阵中的所有元素除以它们的总和。
# 因此，这种情况下需要进行除以 256.0 的操作。
blur = t.autograd.Variable(t.tensor(
          [
            [ 1, 4, 6, 4, 1],
            [ 4,16,24,16, 4],
            [ 6,24,36,24, 6],
            [ 4,16,24,16, 4],
            [ 1, 4, 6, 4, 1]
          ]).to(t.float32)/256.0)
blur.unsqueeze_(0).unsqueeze_(0)
# Sharpen
# Sharpen 滤波器用于增强图像的细节和边缘
sharpen = t.autograd.Variable(t.tensor(
              [
                [ 0,-1, 0],
                [-1, 5,-1],
                [ 0,-1, 0]
              ]).to(t.float32))
sharpen.unsqueeze_(0).unsqueeze_(0)

# Apply convolution for x and y direction
images_b = t.nn.functional.conv2d(images,blur)
images_s = t.nn.functional.conv2d(images_b,sharpen)







# Show images and results
fig, ax = plt.subplots(3, 1, figsize=(8,8))
def imshow(imgs,ax):
  ax.imshow(np.transpose(tv.utils.make_grid(imgs).numpy(), (1, 2, 0)))
imshow(images,ax[0])
imshow(images_b,ax[1])
imshow(images_s,ax[2])
plt.show()
