import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
from SSMs import SSM
from os.path import dirname, join as pjoin
import torch
print(torch.cuda.is_available())
from torch import nn
import time

dtype = torch.float
device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())

plt.close('all')
# Import Data
folderpath = os.getcwd()
filepath = pjoin(folderpath, 'input_3.mat')#'dataset_sysID_3tanks.mat')
data_in = scipy.io.loadmat(filepath)
filepath = pjoin(folderpath, 'output_Q_3.mat')
data_out = scipy.io.loadmat(filepath)


# Extract data from dictionary
dExp, yExp, dExp_val, yExp_val, time__, buildtot, buildtot_val = data_in['dExp'], data_out['yExp'], \
    data_in['dExp_val'], data_out['yExp_val'], data_in['time__'], data_out['buildtotnorm'], data_out['buildtotnorm_val']
nExp = yExp.size

t = time__

t_end = t.size

u = torch.zeros(nExp, t_end, 3, device = device)
y = torch.zeros(nExp, t_end, yExp[0,0].shape[1], device = device)
inputnumberD = 2

for j in range(nExp):
    u[j, :, :] = torch.cat(
            (torch.from_numpy(dExp[0, j][:, 3:]),    # Take columns 3 onwards from dExp
            torch.from_numpy(buildtot[j * t_end : (j + 1) * t_end, 0]).unsqueeze(1)),  # Ensure proper shape for buildtot
            dim=-1
        )    
    y[j, :, :] = (torch.from_numpy(yExp[0, j]))

seed = 55
torch.manual_seed(seed)

idd = 3
hdd = 10
odd = yExp[0, 0].shape[1]

RNN = (SSM
       (#N=3,
        in_features=idd,
        out_features=odd,
        #mid_features=21,
        state_features=hdd,
        scan = True,
        ))
if (torch.cuda.is_available()):
    RNN.cuda()

total_params = sum(p.numel() for p in RNN.parameters())
print(f"Number of parameters: {total_params}")

#RNN = torch.jit.script(RNN)

MSE = nn.MSELoss()

# Define Optimization method
learning_rate = 1.0e-2
optimizer = torch.optim.Adam(RNN.parameters(), lr=learning_rate)
optimizer.zero_grad()

t_end = yExp[0, 0].shape[0]

epochs = 300
LOSS = np.zeros(epochs)

t0= time.time()
for epoch in range(epochs):
    if epoch == epochs - epochs / 2:
        learning_rate = 1.0e-3
        optimizer = torch.optim.Adam(RNN.parameters(), lr=learning_rate)
    if epoch == epochs - epochs / 6:
        learning_rate = 1.0e-4
        optimizer = torch.optim.Adam(RNN.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    loss = 0

    yRNN = RNN(u)
    yRNN = torch.squeeze(yRNN)
    loss = MSE(yRNN, y)
    loss.backward()
    print(u.grad)
    optimizer.step()
    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")
    LOSS[epoch] = loss

t1= time.time()

total_time= t1-t0

nExp = yExp_val.size
t_end = yExp_val[0,0].shape[0]
uval = torch.zeros(nExp, t_end, 3)
yval = torch.zeros(nExp, t_end, 2)

# Fill input and output tensors with validation data
for j in range(nExp):
    uval[j, :, :] = torch.cat(
            (torch.from_numpy(dExp_val[0, j][:, 3:]),    # Take columns 3 onwards from dExp
            torch.from_numpy(buildtot_val[j * t_end : (j + 1) * t_end, 0]).unsqueeze(1)),  # Ensure proper shape for buildtot
            dim=-1
        ) 
    yval[j, :, :] = (torch.from_numpy(yExp_val[0, j]))

yRNN_val = RNN(uval)
yRNN_val = torch.squeeze(yRNN_val)
yval = torch.squeeze(yval)

loss_val = MSE(yRNN_val, yval)

plt.figure('8')
plt.plot(LOSS)
plt.title("LOSS")
plt.show()

plt.figure('9')
plt.plot(yRNN[0, 10000:20000, 0].cpu().detach().numpy(), label='SSM')
plt.plot(y[0, 10000:20000, 0].cpu().detach().numpy(), label='y1 true training set')
plt.title("output 1 train single SSM")
plt.legend()
plt.show()

plt.figure('10')
plt.plot(yRNN_val[:, 0].cpu().detach().numpy(), label='SSM val')
plt.plot(yval[:, 0].cpu().detach().numpy(), label='y1 true validation set')
plt.title("output 1 val single SSM")
plt.legend()
plt.show()

plt.figure('11')
plt.plot(yRNN[0, 10000:20000, 1].cpu().detach().numpy(), label='SSM')
plt.plot(y[0, 10000:20000, 1].cpu().detach().numpy(), label='y2 true training set')
plt.title("output 1 train single SSM")
plt.legend()
plt.show()

plt.figure('12')
plt.plot(yRNN_val[:, 1].cpu().detach().numpy(), label='SSM val')
plt.plot(yval[:, 1].cpu().detach().numpy(), label='y2 true validation set')
plt.title("output 1 val single SSM")
plt.legend()
plt.show()

# plt.figure('15')
# plt.plot(d[inputnumberD, :].detach().numpy(), label='input train')
# plt.plot(dval[inputnumberD, :].detach().numpy(), label='input val')
# plt.title("input single REN")
# plt.legend()
# plt.show()

print(f"Loss Validation single RNN: {loss_val}")