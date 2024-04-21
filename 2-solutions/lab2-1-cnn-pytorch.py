#!/usr/bin/env python3
#
# Simple CNN example to classify images from the Fashion-MNIST dataset.
# This uses pytorch - https://pytorch.org/docs/stable/

import numpy as np
import matplotlib.pyplot as plt

# We use torch for this example and use example images from torchvision
# Install these packages with `pip3 install torch torchvision` or similar
import torch as t
import torchvision as tv

# Get the dataset, split in training and test set
training_data = tv.datasets.FashionMNIST(root="data", train=True, download=True,
                                         transform=tv.transforms.ToTensor())
test_data = tv.datasets.FashionMNIST(root="data", train=False, download=True,
                                     transform=tv.transforms.ToTensor())
# Create data loaders
batch_size = 64
train_dataloader = t.utils.data.DataLoader(training_data, batch_size=batch_size)
test_dataloader = t.utils.data.DataLoader(test_data, batch_size=batch_size)


# Define model
class NeuralNetwork(t.nn.Module):
  def __init__(self):
    super().__init__()
    self.cnn = t.nn.Sequential(
        t.nn.Conv2d(1,32,(3,3)),
        t.nn.ReLU(),
        t.nn.Conv2d(32,32,(3,3)),
        t.nn.ReLU(),
        t.nn.MaxPool2d(2,2),
        t.nn.BatchNorm2d(32),

        t.nn.Conv2d(32,64,(3,3)),
        t.nn.ReLU(),
        t.nn.Conv2d(64,64,(3,3)),
        t.nn.ReLU(),
        t.nn.MaxPool2d(2,2),
        t.nn.BatchNorm2d(64),

        t.nn.Conv2d(64,128,(3,3)),
        t.nn.ReLU(),
        t.nn.MaxPool2d(2,2),
        t.nn.BatchNorm2d(128),

        t.nn.Flatten(1),
        t.nn.Linear(128,256),
        t.nn.Linear(256,10),
        t.nn.Softmax()
      )

  def forward(self, x):
    return self.cnn(x)

# Check if CUDA is available; otherwise train on CPU
device = "cuda" if t.cuda.is_available() else "cpu"
# Setup model
model = NeuralNetwork().to(device)
print(model)

loss_fn = t.nn.CrossEntropyLoss()
optimizer = t.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  model.train() # Tell model we are training it
  # One training step, in batches
  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)
    # Compute prediction error
    pred = model(X)
    loss = loss_fn(pred, y)
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Verbose
    if batch % 100 == 0:
      loss, current = loss.item(), batch * len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  model.eval() # Tell model we are evaluating
  test_loss, correct = 0, 0
  with t.no_grad(): # No gradient, we are only predicting
    # Evaluate in batches
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1) == y).type(t.float).sum().item()
  test_loss /= num_batches
  correct /= size
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 10
for e in range(epochs):
  print(f"Epoch {e+1}\n-------------------------------")
  train(train_dataloader, model, loss_fn, optimizer)
  test(test_dataloader, model, loss_fn)