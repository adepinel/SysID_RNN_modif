# Import necessary libraries
from src.models import RNNModel
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
from os.path import dirname, join as pjoin
import torch
from torch import nn

# Set data type and device
dtype = torch.float
device = torch.device("cpu")

# Close all open plots
plt.close('all')

# Load data from file
folderpath = os.getcwd()
filepath = pjoin(folderpath, 'data/dataset_sysID_3tanks_final.mat')
data = scipy.io.loadmat(filepath)

# Extract data from dictionary
dExp, yExp, dExp_val, yExp_val, Ts = data['dExp'], data['yExp'], \
    data['dExp_val'], data['yExp_val'], data['Ts'].item()
nExp = yExp.size

# Create time vector
t = np.arange(0, np.size(dExp[0, 0], 1) * Ts, Ts)

# Set end time
t_end = t.size

# Initialize input and output tensors
u = torch.zeros(nExp, t_end, 1)
y = torch.zeros(nExp, t_end, 3)

# Set input number
inputnumberD = 1

# Fill input and output tensors with data
for j in range(nExp):
    inputActive = (torch.from_numpy(dExp[0, j])).T
    u[j, :, :] = torch.unsqueeze(inputActive[:,inputnumberD], 1)
    y[j, :, :] = (torch.from_numpy(yExp[0, j])).T

# Set random seed for reproducibility
seed = 1
torch.manual_seed(seed)

# Set dimensions for RNN layers
idd = 1
hdd = 20
ldd = 5
odd = yExp[0, 0].shape[0]

# Initialize RNN model
RNN = RNNModel(idd, hdd, ldd, odd)

# Define loss function
MSE = nn.MSELoss()

# Define optimization method
learning_rate = 1.0e-2
optimizer = torch.optim.Adam(RNN.parameters(), lr=learning_rate)
optimizer.zero_grad()

# Set end time for training data
t_end = yExp[0, 0].shape[1]

# Set number of epochs
epochs = 200

# Initialize array to store loss values
LOSS = np.zeros(epochs)

# Train the RNN model
for epoch in range(epochs):
    # Adjust learning rate at certain epochs
    if epoch == epochs - epochs / 2:
        learning_rate = 1.0e-3
        optimizer = torch.optim.Adam(RNN.parameters(), lr=learning_rate)
    if epoch == epochs - epochs / 6:
        learning_rate = 1.0e-4
        optimizer = torch.optim.Adam(RNN.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    loss = 0

    # Get RNN output for training data
    yRNN = RNN(u)
    yRNN = torch.squeeze(yRNN)

    # Calculate loss and backpropagate
    loss = MSE(yRNN, y)
    loss.backward()
    optimizer.step()

    # Print loss for current epoch
    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")
    LOSS[epoch] = loss

# Set end time for validation data
t_end = yExp_val[0, 0].shape[1]

# Initialize input and output tensors for validation data
nExp = yExp_val.size
uval = torch.zeros(nExp, t_end, 1)
yval = torch.zeros(nExp, t_end, 3)

# Fill input and output tensors with validation data
for j in range(nExp):
    inputActive = (torch.from_numpy(dExp_val[0, j])).T
    uval[j, :, :] = torch.unsqueeze(inputActive[:,inputnumberD], 1)
    yval[j, :, :] = (torch.from_numpy(yExp_val[0, j])).T

# Get RNN output for validation data
yRNN_val = RNN(uval)
yRNN_val = yRNN_val.unsqueeze_(0)

# Calculate loss for validation data
loss_val = MSE(yRNN_val, yval)

# Plot loss over epochs
plt.figure('8')
plt.plot(LOSS)
plt.title("LOSS")
plt.show()

# Plot output 1 for training data
plt.figure('9')
plt.plot(yRNN[0, :, 0].detach().numpy(), label='REN')
plt.plot(y[0, :, 0].detach().numpy(), label='y train')
plt.title("output 1 train single RNN")
plt.legend()
plt.show()

# Plot output 1 for validation data
plt.figure('10')
plt.plot(yRNN_val[:, 0].detach().numpy(), label='REN val')
plt.plot(yval[0, :, 0].detach().numpy(), label='y val')
plt.title("output 1 val single RNN")
plt.legend()
plt.show()

# Plot output 2 for training data
plt.figure('11')
plt.plot(yRNN[0, :, 1].detach().numpy(), label='REN')
plt.plot(y[0, :, 1].detach().numpy(), label='y train')
plt.title("output 2 train single RNN")
plt.legend()
plt.show()

# Plot output 2 for validation data
plt.figure('12')
plt.plot(yRNN_val[0, :, 1].detach().numpy(), label='REN val')
plt.plot(yval[0, :, 1].detach().numpy(), label='y val')
plt.title("output 2 val single REN")
plt.legend()
plt.show()

# Plot output 3 for training data
plt.figure('13')
plt.plot(yRNN[0, :, 2].detach().numpy(), label='REN')
plt.plot(y[0, :, 2].detach().numpy(), label='y train')
plt.title("output 3 train single RNN")
plt.legend()
plt.show()

# Plot output 3 for validation data
plt.figure('14')
plt.plot(yRNN_val[0,:, 2].detach().numpy(), label='REN val')
plt.plot(yval[0, :, 2].detach().numpy(), label='y val')
plt.title("output 3 val single RNN")
plt.legend()
plt.show()

# Save RNN output and validation data to file
scipy.io.savemat('data_singleRNN.mat', dict(yRENm_val=yRNN_val.detach().numpy(), yval=yval.detach().numpy()))

# Print validation loss
print(f"Loss Validation single RNN: {loss_val}")
