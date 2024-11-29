#%% Packages
from models import NetworkedRENs, REN, RNNModel
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os

#%% Init
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from os.path import dirname, join as pjoin
import torch
from torch import nn

print(torch.__version__)  # Check PyTorch version
print(torch.version.cuda)  # Check CUDA version PyTorch was compiled with

dtype = torch.float
device = torch.device("cuda")
print(device)
print(torch.cuda.is_available())

#%% Import Data
folderpath = os.getcwd()
filepath = pjoin(folderpath, 'input_3.mat')#'dataset_sysID_3tanks.mat')
data_in = scipy.io.loadmat(filepath)
filepath = pjoin(folderpath, 'output_Q_3.mat')
data_out = scipy.io.loadmat(filepath)
filepath = pjoin(folderpath, 'subsystems.mat')
data_sub = scipy.io.loadmat(filepath)

filepath = pjoin(folderpath, 'denormalize.mat')
data_max = scipy.io.loadmat(filepath)

#%% Extract data from dictionary
maxTrit, maxTdel = data_max['maxTrit'], data_max['maxTman']
Toutass_t, Toutass_v, Toutchill_t, Toutchill_v = data_sub['Toutass_train'], data_sub['Toutass_val'], data_sub['Toutchillers_train'], data_sub['Toutchillers_val']
dExp, yExp, dExp_val, yExp_val, time__, buildtot, buildtot_val = data_in['dExp'], data_out['yExp'], \
    data_in['dExp_val'], data_out['yExp_val'], data_in['time__'], data_out['buildtotnorm'], data_out['buildtotnorm_val']
nExp = yExp.size

t = time__

t_end = t.size

#%% initialize the model

ny = np.shape(yExp[0,-1])[1]
nd = np.shape(dExp[0,-1])[1]

#t = np.arange(0, np.size(dExp[0, 0], 1) * Ts-Ts, Ts)
#t_end = yExp[0, 0].shape[1] - 1

for exp in range(nExp):
    y_exp = yExp[0,exp]
    d_exp = dExp[0,exp]
    plt.figure(figsize=(4 * 2, 4))
    for out in range(ny):
        plt.subplot(2, 2, out+1)
        plt.plot(t[14000:16500], y_exp[14000:16500,out])
        plt.title(r"Water level h%i "%out + r"in experiment %i"%(exp+1))
    plt.subplot(2, 2, ny+1)
    plt.plot(t[14000:16500], d_exp[14000:16500,1])
    plt.title(r"v in experiment %i"%(exp+1))
    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.4)
plt.show()


#%% TRAIN OF NETWORKED RENs
epochs = 150
t_end = 2500

torch.manual_seed(2)
N = 3 # Number of interconnected systems

n = torch.tensor([2, 2, 2])  # input dimensions
p = torch.tensor([1, 1, 1])  # output dimensions

n_xi = np.array([5, 5, 5]) # nel paper n1, numero di stati
l = np.array([5, 5, 5])  # nel paper q, dimension of the square matrix D11 -- number of _non-linear layers_ of the RE

#alpha = 0.6
#beta = 0.4

#Muy = torch.cat((torch.tensor([[0, alpha, beta], [1, 0, 0], [1, 0, 0]]), torch.zeros(3,3)), dim=0)
#Muy = Muy.float()

Mud = torch.cat((torch.zeros(3,3), torch.eye(3)), dim=0)
#Mey = torch.tensor([[0, alpha, beta], [1, 0, 0]])

# Define the system
RENsys = NetworkedRENs(N, Mud, n, p, n_xi, l)

# Define Loss function
MSE = nn.MSELoss()

# Define Optimization method
learning_rate = 1.0e-1
optimizer = torch.optim.Adam(RENsys.parameters(), lr=learning_rate)
optimizer.zero_grad()

LOSS = np.zeros(epochs)
loss = 0

for epoch in range(epochs):
    if epoch == epochs - epochs / 2:
        learning_rate = 1.0e-2
        optimizer = torch.optim.Adam(RENsys.parameters(), lr=learning_rate)
    if epoch == epochs - epochs / 6:
        learning_rate = 1.0e-3
        optimizer = torch.optim.Adam(RENsys.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    loss = 0
    for exp in range(nExp - 1):
        xi = []
        y = torch.cat((torch.from_numpy(yExp[0, exp][14000:16500,0]).float().to(device).unsqueeze(1),
                       torch.from_numpy(Toutass_t[exp*30240 +14000:exp*30240 + 16500]).float().to(device),
                       torch.from_numpy(Toutchill_t[exp*30240 +14000:exp*30240 + 16500]).float().to(device)), dim=1)
        y = y.T
        yRENm = torch.randn(3,t_end , device=device, dtype=dtype)
        yRENm[0,:] = y[0,:]
        for j in range(N):
            xi.append(torch.randn(RENsys.r[j].n, device=device, dtype=dtype))
        d = torch.cat((torch.from_numpy(buildtot[exp*30240 +14000:exp*30240 + 16500]).float().to(device),
                       torch.from_numpy(dExp[0, exp][14000:16500,-1]).float().to(device).unsqueeze(1),
                       torch.from_numpy(dExp[0, exp][14000:16500,-2]).float().to(device).unsqueeze(1)), dim=1)
        d = d.T
        xi = torch.cat(xi)
        for t in range(1, t_end):
            yRENm[:, t], xi = RENsys(t, d[:, t - 1], xi)

        loss = loss + MSE(yRENm[:, 0:yRENm.size(1)], y[:, 0:t_end + 1])
        # ignorare da loss effetto condizione iniziale

    loss = loss / nExp
    loss.backward()
    # loss.backward(retain_graph=True)

    optimizer.step()

    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")
    for net in range(N):
        print(f"L2 gain REN%i"%net+":%.1f"%RENsys.r[net].gamma)
    LOSS[epoch] = loss
    
    
    # save a checkpoint with optimizer and other details
    torch.save({
        'epoch': epoch,
        'model_state_dict': RENsys.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f'gpu_epoch_{epoch+1}.pth')

# Validation for Networked RENs
torch.manual_seed(2)  # Ensure reproducibility

# Validation parameters
t_end = 2500  # Same as training sequence length
xi = []


# Prepare validation outputs
yval = torch.cat((
    torch.from_numpy(yExp_val[0, 0][2000:4500, 0]).float().to(device).unsqueeze(1),
    torch.from_numpy(Toutass_v[2000:4500]).float().to(device),#.unsqueeze(1),
    torch.from_numpy(Toutchill_v[2000:4500]).float().to(device)#.unsqueeze(1)
), dim=1).T

yRENm_val = torch.zeros(3, t_end, device=device, dtype=dtype)
yRENm_val[0, :] = yval[0, :]  # Initialize with first channel of validation output

# Initialize states for REN validation
for j in range(N):
    xi.append(torch.randn(RENsys.r[j].n, device=device, dtype=dtype))
xi = torch.cat(xi)

print(buildtot_val[2000:4500].shape)
print(dExp_val[0, 0][2000:4500, -1].shape)
print(dExp_val[0, 0][2000:4500, -2].shape)
# Prepare validation disturbances
dval = torch.cat((
    torch.from_numpy(buildtot_val[2000:4500]).float().to(device),#.unsqueeze(0),
    torch.from_numpy(dExp_val[0, 0][2000:4500, -1]).float().to(device).unsqueeze(1),
    torch.from_numpy(dExp_val[0, 0][2000:4500, -2]).float().to(device).unsqueeze(1)
), dim=1)

print(dval.size())
dval = dval.T

# Validation simulation loop
loss_val = 0
for t in range(1, t_end):
    yRENm_val[:, t], xi = RENsys(t, dval[:, t - 1], xi)

# Compute validation loss
loss_val = MSE(yRENm_val[:, :t_end], yval[:, :t_end])

# Plot Training Outputs
plt.figure(figsize=(8, 4))
for out in range(3):  # Assuming 3 outputs
    plt.subplot(1, 3, out + 1)
    plt.plot(yRENm_val[out, :].detach().cpu().numpy(), label="Networked REN Validation")
    plt.plot(yval[out, :].detach().cpu().numpy(), label="Validation Target")
    plt.title(f"Output h{out} in Validation")
    plt.legend()
plt.tight_layout()
plt.show()

# Total Parameters in Networked RENs
total_params = sum(p.numel() for p in RENsys.parameters() if p.requires_grad)

# Print Validation Results
print(f"Validation Loss: {loss_val.item()}")
print(f"Total trainable parameters in REN system: {total_params}")

# %%
