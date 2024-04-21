#!/usr/bin/env python3
#
# Autoencoder in tensorflow

import numpy as np
import matplotlib.pyplot as plt

# Use tensorflow's keras (needs tensorflow or other supported backend; GPU version highly recommended)
# Install tensorflow with `pipe3 install tensorflow` or similar
import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization

# Get the dataset, split in training and test set
# x is the image, y the classification
(train_x,_), (test_x,_) = k.datasets.fashion_mnist.load_data()

# Preprocess images
# Normalise
train_x = train_x.astype('float32') / 255.0
test_x = test_x.astype('float32') / 255.0
# Add channel axis
train_x = train_x.reshape(-1, 28,28, 1)
test_x = test_x.reshape(-1, 28,28, 1)
 # Rsize to 32 as it avoids down/upscaling issue on dimensions in autoencoder
train_x = tf.image.resize(train_x, [32,32])
test_x = tf.image.resize(test_x, [32,32])
# Add noise to input
train_x_noisy = train_x + 0.1 * np.random.normal(loc=0.0, scale=1.0, size=train_x.shape) 
test_x_noisy = test_x + 0.1 * np.random.normal(loc=0.0, scale=1.0, size=test_x.shape) 

# Simple autoencoder
# We use the tensorflow functional interface (Sequential would also work)
input = k.layers.Input(shape=(32,32,1))

# Encoder

x = Conv2D(16, (3,3), activation="relu", padding="same")(input)
x = MaxPooling2D(pool_size=(2,2))(x)
x = BatchNormalization()(x)

x = Conv2D(8, (3,3), activation="relu", padding="same")(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = BatchNormalization()(x)

x = Conv2D(8, (3,3), activation="relu", padding="same")(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = BatchNormalization()(x)

# Decoder

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = BatchNormalization()(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = BatchNormalization()(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = BatchNormalization()(x)

x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)

# Autoencoder model
model = Model(input, x)

# Create and fit model using crossentropy loss
epochs = 10 # 10 epochs for testing; try 50 for better results
model.compile(loss=k.losses.MeanAbsoluteError(),
              optimizer=k.optimizers.Adam(), metrics=['accuracy'])
# Save figure of model
k.utils.plot_model(model, to_file=("lab2-3-autoencoder.png"), show_shapes=True, 
                   show_dtype=True, show_layer_names=True, show_layer_activations=True,
                   rankdir='TB', expand_nested=True, dpi=96)
# Train model with input and output being the noisy and clean image for noise filtering
h = model.fit(train_x_noisy, train_x, batch_size=32, epochs=epochs,
              validation_data=(test_x_noisy, test_x))

# Evaluate model on test set
test_loss, test_acc = model.evaluate(test_x, test_x)
print('Test loss', test_loss)
print('Test accuracy', test_acc)

# Plot training history for training and validation accuracy
fig, ax = plt.subplots(3, 1, figsize=(8,8))
ax[0].plot(h.history['accuracy'])
ax[0].plot(h.history['val_accuracy'])
ax[0].set_title('Model Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend(['train', 'test'], loc='upper left')

# Run prediction, report result for first image and show that image
pred = model.predict(test_x[0:1,...])
ax[1].imshow(test_x[0,:,:,0], cmap="gray", vmin=0, vmax=1)
ax[2].imshow(pred[0,:,:,0], cmap="gray", vmin=0, vmax=1)
plt.show()