#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:24:52 2023

@author: Danilo Saccani (danilo.saccani@epfl.ch)
"""

import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


dtype = torch.float
#device = torch.device("cpu")
device = "cuda" #if torch.cuda.is_available() else "cpu"

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, nonlinearity='tanh', batch_first=True, bias=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state randomly
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out)
        out = out.squeeze()
        return out


class LSTModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTModel, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # RNN
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True,bias=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # One time step
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out)
        # out = out.squeeze()
        return out
"""
# chatgpt correction...
# REN WITH TRAINABLE GAMMA
class RENR(nn.Module):
    def __init__(self, n, m, n_xi, l):
        super().__init__()
        self.n = n  # nel paper m input
        self.n_xi = n_xi  # nel paper n1 w hdd
        self.l = l  # nel paper q x
        self.m = m  # nel paper p1 output
        self.s = np.max((n, m))  # s nel paper, dimensione di X3 Y3

        # # # # # # # # # Training parameters # # # # # # # # #
        std = 1
        self.X = nn.Parameter(torch.randn(2 * n_xi + l, 2 * n_xi + l) * std)
        self.Y = nn.Parameter(torch.randn(n_xi, n_xi) * std)
        self.B2 = nn.Parameter(torch.randn(n_xi, n) * std)
        self.C2 = nn.Parameter(torch.randn(m, n_xi) * std).to(device)
        self.D21 = nn.Parameter(torch.randn(m, l) * std).to(device)
        self.X3 = nn.Parameter(torch.randn(self.s, self.s) * std)
        self.Y3 = nn.Parameter(torch.randn(self.s, self.s) * std)
        self.sg = nn.Parameter(torch.randn(1, 1) * std)

        # v signal
        self.D12 = nn.Parameter(torch.randn(l, n) * std)
        self.epsilon = 0.001
        self.set_model_param()  # Initialize auxiliary parameters

    def set_model_param(self):
        n_xi, l, n, m = self.n_xi, self.l, self.n, self.m
        gamma = self.sg ** 2

        # Auxiliary calculations
        R = gamma * torch.eye(n, n)
        Q = (-1 / gamma) * torch.eye(m, m)
        M = F.linear(self.X3, self.X3) + self.Y3 - self.Y3.T + self.epsilon * torch.eye(self.s)
        M_tilde = F.linear(torch.eye(self.s) - M.clone(), torch.inverse(torch.eye(self.s) + M.clone()).T)
        Zeta = M_tilde[:self.m, :self.n]
        
        # Cloning tensors to avoid inplace modification issues
        self.D22 = (gamma * Zeta).clone()
        R_capital = (R - (1 / gamma) * F.linear(self.D22.T, self.D22.T)).clone()
        C2_capital = (torch.matmul(torch.matmul(self.D22.T, Q), self.C2)).clone()
        D21_capital = (torch.matmul(torch.matmul(self.D22.T, Q), self.D21) - self.D12.T).clone()

        vec_R = torch.cat([C2_capital.T, D21_capital.T, self.B2], 0).clone()
        vec_Q = torch.cat([self.C2.T, self.D21.T, torch.zeros(n_xi, m)], 0).clone()

        H = (torch.matmul(self.X.T, self.X) + self.epsilon * torch.eye(2 * n_xi + l) +
             torch.matmul(torch.matmul(vec_R, torch.inverse(R_capital)), vec_R.T) -
             torch.matmul(torch.matmul(vec_Q, Q), vec_Q.T)).clone()

        # Split H and clone to avoid inplace modification errors
        h1, h2, h3 = torch.split(H, (n_xi, l, n_xi), dim=0)
        H11, H12, H13 = torch.split(h1, (n_xi, l, n_xi), dim=1)
        H21, H22, _ = torch.split(h2, (n_xi, l, n_xi), dim=1)
        H31, H32, H33 = torch.split(h3, (n_xi, l, n_xi), dim=1)

        # Set parameters while ensuring they aren't modified in-place
        self.F = H31.clone()
        self.B1 = H32.clone()
        self.E = 0.5 * (H11.clone() + H33.clone() + self.Y - self.Y.T)
        self.Lambda = torch.diag(H22.clone())
        self.D11 = -torch.tril(H22.clone(), diagonal=-1)
        self.C1 = -H21.clone()

    def forward(self, t, w, xi):
        assert xi.device == device, f"xi is on {xi.device}, expected {device}"
        assert w.device == device, f"w is on {w.device}, expected {device}"
        print(f"xi: {xi.device}, w: {w.device}")
        vec = torch.zeros(self.l, device=xi.device)
        vec[0] = 1
        epsilon = torch.zeros(self.l, device=xi.device)

        v = F.linear(xi, self.C1[0, :]) + F.linear(w, self.D12[0, :])
        epsilon = epsilon + vec * torch.tanh(v / self.Lambda[0])
        
        for i in range(1, self.l):
            vec = torch.zeros(self.l, device=xi.device)
            vec[i] = 1
            v = F.linear(xi, self.C1[i, :]) + F.linear(epsilon, self.D11[i, :]) + F.linear(w, self.D12[i, :])
            epsilon = epsilon + vec * torch.tanh(v / self.Lambda[i])
        
        E_xi_ = F.linear(xi, self.F) + F.linear(epsilon, self.B1) + F.linear(w, self.B2)
        xi_ = F.linear(E_xi_, torch.inverse(self.E))
        u = F.linear(xi, self.C2) + F.linear(epsilon, self.D21) + F.linear(w, self.D22)

        return u, xi_


"""
# REN WITH TRAINABLE GAMMA
class RENR(nn.Module):
    def __init__(self, n, m, n_xi, l):
        super().__init__()
        self.n = n  # nel paper m input
        self.n_xi = n_xi  # nel paper n1 w hdd 
        self.l = l  # nel paper q x
        self.m = m  # nel paper p1 output
        self.s = np.max((n, m))  # s nel paper, dimensione di X3 Y3

        # # # # # # # # # Training parameters # # # # # # # # #
        # Auxiliary matrices:
        std = 1
        self.X = nn.Parameter((torch.randn(2 * n_xi + l, 2 * n_xi + l) * std)).to(device)
        self.Y = nn.Parameter((torch.randn(n_xi, n_xi) * std)).to(device)  # Y1 nel paper
        # NN state dynamics:
        self.B2 = nn.Parameter((torch.randn(n_xi, n) * std)).to(device)
        # NN output:
        self.C2 = nn.Parameter((torch.randn(m, n_xi) * std)).to(device)
        self.D21 = nn.Parameter((torch.randn(m, l) * std)).to(device)
        self.X3 = nn.Parameter(torch.randn(self.s, self.s) * std).to(device)
        self.Y3 = nn.Parameter(torch.randn(self.s, self.s) * std).to(device)
        self.sg = nn.Parameter(torch.randn(1, 1) * std).to(device)  # square root of gamma

        # v signal:
        self.D12 = nn.Parameter((torch.randn(l, n) * std)).to(device)
        # bias:
        # self.bxi = nn.Parameter(torch.randn(n_xi))
        # self.bv = nn.Parameter(torch.randn(l))
        # self.bu = nn.Parameter(torch.randn(m))
        # # # # # # # # # Non-trainable parameters # # # # # # # # #
        # Auxiliary elements
        self.epsilon = 0.001
        self.F = torch.zeros(n_xi, n_xi).to(device)
        self.B1 = torch.zeros(n_xi, l).to(device)
        self.E = torch.zeros(n_xi, n_xi).to(device)
        self.Lambda = torch.ones(l).to(device)
        self.C1 = torch.zeros(l, n_xi).to(device)
        self.D11 = torch.zeros(l, l).to(device)
        self.Lq = torch.zeros(m, m).to(device)
        self.Lr = torch.zeros(n, n).to(device)
        self.D22 = torch.zeros(m, n).to(device)
        self.set_model_param()

    def set_model_param(self):
        n_xi = self.n_xi
        l = self.l
        n = self.n
        m = self.m
        gamma = self.sg ** 2
        R = gamma * torch.eye(n, n)
        Q = (-1 / gamma) * torch.eye(m, m)
        M = F.linear(self.X3, self.X3) + self.Y3 - self.Y3.T + self.epsilon * torch.eye(self.s)
        M_tilde = F.linear(torch.eye(self.s) - M,
                           torch.inverse(torch.eye(self.s) + M).T)
        Zeta = M_tilde[0:self.m, 0:self.n]
        self.D22 = gamma * Zeta
        R_capital = R - (1 / gamma) * F.linear(self.D22.T, self.D22.T)
        C2_capital = torch.matmul(torch.matmul(self.D22.T, Q), self.C2)
        D21_capital = torch.matmul(torch.matmul(self.D22.T, Q), self.D21) - self.D12.T
        vec_R = torch.cat([C2_capital.T, D21_capital.T, self.B2], 0)
        vec_Q = torch.cat([self.C2.T, self.D21.T, torch.zeros(n_xi, m)], 0)
        H = torch.matmul(self.X.T, self.X) + self.epsilon * torch.eye(2 * n_xi + l) + torch.matmul(
            torch.matmul(vec_R, torch.inverse(R_capital)), vec_R.T) - torch.matmul(
            torch.matmul(vec_Q, Q), vec_Q.T)
        h1, h2, h3 = torch.split(H, (n_xi, l, n_xi), dim=0)
        H11, H12, H13 = torch.split(h1, (n_xi, l, n_xi), dim=1)
        H21, H22, _ = torch.split(h2, (n_xi, l, n_xi), dim=1)
        H31, H32, H33 = torch.split(h3, (n_xi, l, n_xi), dim=1)
        P = H33
        # NN state dynamics:
        self.F = H31
        self.B1 = H32
        # NN output:
        self.E = 0.5 * (H11 + P + self.Y - self.Y.T)
        # v signal:  [Change the following 2 lines if we don't want a strictly acyclic REN!]
        self.Lambda = torch.diag(H22)
        self.D11 = -torch.tril(H22, diagonal=-1)
        self.C1 = -H21

    def forward(self, t, w, xi):
        vec = torch.zeros(self.l)
        vec[0] = 1
        epsilon = torch.zeros(self.l)
        v = F.linear(xi.to(device), self.C1[0, :].to(device)) + F.linear(w.to(device),
                                                   self.D12[0, :].to(device))  # + self.bv[0]
        epsilon = epsilon.to(device) + vec * torch.tanh(v / self.Lambda[0].to(device))
        for i in range(1, self.l):
            vec = torch.zeros(self.l)
            vec[i] = 1
            v = F.linear(xi, self.C1[i, :]) + F.linear(epsilon,
                                                       self.D11[i, :]) + F.linear(w, self.D12[i, :])  # self.bv[i]
            epsilon = epsilon + vec * torch.tanh(v / self.Lambda[i])
        E_xi_ = F.linear(xi, self.F) + F.linear(epsilon,
                                                self.B1) + F.linear(w, self.B2)  # + self.bxi
        xi_ = F.linear(E_xi_, self.E.inverse())
        u = F.linear(xi, self.C2) + F.linear(epsilon, self.D21) + \
            F.linear(w, self.D22)  # + self.bu
        return u, xi_


# REN implementation in the acyclic version
# See paper: "Recurrent Equilibrium Networks: Flexible dynamic models with guaranteed stability and robustness"
class RenG(nn.Module):
    # ## Implementation of REN model, modified from "Recurrent Equilibrium Networks: Flexible Dynamic Models with
    # Guaranteed Stability and Robustness" by Max Revay et al.
    def __init__(self, m, p, n, l, bias=False, mode="l2stable", gamma=0.3, Q=None, R=None, S=None,
                 device=torch.device('cpu')):
        super().__init__()
        self.m = m  # input dimension
        self.n = n  # state dimension
        self.l = l  # dimension of v(t) and w(t)
        self.p = p  # output dimension
        self.mode = mode
        self.device = device
        self.gamma = gamma
        # # # # # # # # # IQC specification # # # # # # # # #
        self.Q = Q
        self.R = R
        self.S = S
        # # # # # # # # # Training parameters # # # # # # # # #
        # Auxiliary matrices:
        std = 1

        # Sparse training matrix parameters
        self.x0 = nn.Parameter((torch.randn(1, n, device=device) * std))
        self.X = nn.Parameter((torch.randn(2 * n + l, 2 * n + l, device=device) * std))
        self.Y = nn.Parameter((torch.randn(n, n, device=device) * std))
        self.Z3 = nn.Parameter(torch.randn(abs(p - m), min(p, m), device=device) * std)
        self.X3 = nn.Parameter(torch.randn(min(p, m), min(p, m), device=device) * std)
        self.Y3 = nn.Parameter(torch.randn(min(p, m), min(p, m), device=device) * std)
        self.D12 = nn.Parameter(torch.randn(l, m, device=device))
        #self.D21 = nn.Parameter((torch.randn(p, l, device=device) * std))
        self.B2 = nn.Parameter((torch.randn(n, m, device=device) * std))
        self.C2 = nn.Parameter((torch.randn(p, n, device=device) * std))

        if bias:
            self.bx = nn.Parameter(torch.randn(n, device=device) * std)
            self.bv = nn.Parameter(torch.randn(l, device=device) * std)
            self.bu = nn.Parameter(torch.randn(p, device=device) * std)
        else:
            self.bx = torch.zeros(n, device=device)
            self.bv = torch.zeros(l, device=device)
            self.bu = torch.zeros(p, device=device)
        # # # # # # # # # Non-trainable parameters # # # # # # # # #
        # Auxiliary elements

        self.x = torch.zeros(1, n, device=device)
        self.epsilon = 0.001
        self.F = torch.zeros(n, n, device=device)
        self.B1 = torch.zeros(n, l, device=device)
        self.E = torch.zeros(n, n, device=device)
        self.Lambda = torch.ones(l, device=device)
        self.C1 = torch.zeros(l, n, device=device)
        self.D11 = torch.zeros(l, l, device=device)
        self.D22 = torch.zeros(p, m, device=device)
        self.P = torch.zeros(n, n, device=device)
        self.P_cal = torch.zeros(n, n, device=device)
        self.D21 = torch.zeros(p, l, device=device)
        self.set_param(gamma)

    def set_param(self, gamma=0.3):
        n, l, m, p = self.n, self.l, self.m, self.p
        self.Q, self.R, self.S = self._set_mode(self.mode, gamma, self.Q, self.R, self.S)
        M = F.linear(self.X3.T, self.X3.T) + self.Y3 - self.Y3.T + F.linear(self.Z3.T,
                                                                            self.Z3.T) + self.epsilon * torch.eye(
            min(m, p), device=self.device)
        if p >= m:
            N = torch.vstack((F.linear(torch.eye(m, device=self.device) - M,
                                       torch.inverse(torch.eye(m, device=self.device) + M).T),
                              -2 * F.linear(self.Z3, torch.inverse(torch.eye(m, device=self.device) + M).T)))
        else:
            N = torch.hstack((F.linear(torch.inverse(torch.eye(p, device=self.device) + M),
                                       (torch.eye(p, device=self.device) - M).T),
                              -2 * F.linear(torch.inverse(torch.eye(p, device=self.device) + M), self.Z3)))

        Lq = torch.linalg.cholesky(-self.Q).T
        Lr = torch.linalg.cholesky(self.R - torch.matmul(self.S, torch.matmul(torch.inverse(self.Q), self.S.T))).T
        self.D22 = -torch.matmul(torch.inverse(self.Q), self.S.T) + torch.matmul(torch.inverse(Lq),
                                                                                 torch.matmul(N, Lr))
        # Calculate psi_r:
        R_cal = self.R + torch.matmul(self.S, self.D22) + torch.matmul(self.S, self.D22).T + torch.matmul(self.D22.T,
                                                                                                          torch.matmul(
                                                                                                              self.Q,
                                                                                                              self.D22))
        R_cal_inv = torch.linalg.inv(R_cal)
        C2_cal = torch.matmul(torch.matmul(self.D22.T, self.Q) + self.S, self.C2).T
        D21_cal = torch.matmul(torch.matmul(self.D22.T, self.Q) + self.S, self.D21).T - self.D12
        vec_r = torch.cat((C2_cal, D21_cal, self.B2), dim=0)
        psi_r = torch.matmul(vec_r, torch.matmul(R_cal_inv, vec_r.T))
        # Calculate psi_q:
        vec_q = torch.cat((self.C2.T, self.D21.T, torch.zeros(self.n, self.p, device=self.device)), dim=0)
        psi_q = torch.matmul(vec_q, torch.matmul(self.Q, vec_q.T))
        # Create H matrix:
        H = torch.matmul(self.X.T, self.X) + self.epsilon * torch.eye(2 * n + l, device=self.device) + psi_r - psi_q
        h1, h2, h3 = torch.split(H, [n, l, n], dim=0)
        H11, H12, H13 = torch.split(h1, [n, l, n], dim=1)
        H21, H22, _ = torch.split(h2, [n, l, n], dim=1)
        H31, H32, H33 = torch.split(h3, [n, l, n], dim=1)
        self.P_cal = H33
        # NN state dynamics:
        self.F = H31
        self.B1 = H32
        # NN output:
        self.E = 0.5 * (H11 + self.P_cal + self.Y - self.Y.T)
        # v signal:  [Change the following 2 lines if we don't want a strictly acyclic REN!]
        self.Lambda = 0.5 * torch.diag(H22)
        self.D11 = -torch.tril(H22, diagonal=-1)
        self.C1 = -H21
        # Matrix P
        self.P = torch.matmul(self.E.T, torch.matmul(torch.inverse(self.P_cal), self.E))

    def forward(self, t, u, x):
        decay_rate = 0.95
        vec = torch.zeros(self.l, device=self.device)
        epsilon = torch.zeros(self.l, device=self.device)
        if self.l > 0:
            vec[0] = 1
            v = F.linear(x, self.C1[0, :]) + F.linear(u, self.D12[0, :]) + (decay_rate ** t) * self.bv[0]
            epsilon = epsilon + vec * torch.tanh(v / self.Lambda[0])
        for i in range(1, self.l):
            vec = torch.zeros(self.l, device=self.device)
            vec[i] = 1
            v = F.linear(x, self.C1[i, :]) + F.linear(epsilon, self.D11[i, :]) + F.linear(u, self.D12[i, :]) + (
                    decay_rate ** t) * self.bv[i]
            epsilon = epsilon + vec * torch.tanh(v / self.Lambda[i])
        E_x_ = F.linear(x, self.F) + F.linear(epsilon, self.B1) + F.linear(u, self.B2) + (decay_rate ** t) * self.bx

        x_ = F.linear(E_x_, self.E.inverse())

        y = F.linear(x, self.C2) + F.linear(epsilon, self.D21) + F.linear(u, self.D22) + (decay_rate ** t) * self.bu

        return y, x_

    def _set_mode(self, mode, gamma, Q, R, S, eps=1e-4):
        # We set Q to be negative definite. If Q is nsd we set: Q - \epsilon I.
        # I.e. The Q we define here is denoted as \matcal{Q} in REN paper.
        if mode == "l2stable":
            Q = -(1. / gamma) * torch.eye(self.p, device=self.device)
            R = gamma * torch.eye(self.m, device=self.device)
            S = torch.zeros(self.m, self.p, device=self.device)
        elif mode == "input_p":
            if self.p != self.m:
                raise NameError("Dimensions of u(t) and y(t) need to be the same for enforcing input passivity.")
            Q = torch.zeros(self.p, self.p, device=self.device) - eps * torch.eye(self.p, device=self.device)
            R = -2. * gamma * torch.eye(self.m, device=self.device)
            S = torch.eye(self.p, device=self.device)
        elif mode == "output_p":
            if self.p != self.m:
                raise NameError("Dimensions of u(t) and y(t) need to be the same for enforcing output passivity.")
            Q = -2. * gamma * torch.eye(self.p, device=self.device)
            R = torch.zeros(self.m, self.m, device=self.device)
            S = torch.eye(self.m, device=self.device)
        else:
            print("Using matrices R,Q,S given by user.")
            # Check dimensions:
            if not (len(R.shape) == 2 and R.shape[0] == R.shape[1] and R.shape[0] == self.m):
                raise NameError("The matrix R is not valid. It must be a square matrix of %ix%i." % (self.m, self.m))
            if not (len(Q.shape) == 2 and Q.shape[0] == Q.shape[1] and Q.shape[0] == self.p):
                raise NameError("The matrix Q is not valid. It must be a square matrix of %ix%i." % (self.p, self.p))
            if not (len(S.shape) == 2 and S.shape[0] == self.m and S.shape[1] == self.p):
                raise NameError("The matrix S is not valid. It must be a matrix of %ix%i." % (self.m, self.p))
            # Check R=R':
            if not (R == R.T).prod():
                raise NameError("The matrix R is not valid. It must be symmetric.")
            # Check Q is nsd:
            eigs, _ = torch.linalg.eig(Q)
            if not (eigs.real <= 0).prod():
                print('oh!')
                raise NameError("The matrix Q is not valid. It must be negative semidefinite.")
            if not (eigs.real < 0).prod():
                # We make Q negative definite: (\mathcal{Q} in the REN paper)
                Q = Q - eps * torch.eye(self.p, device=self.device)
        return Q, R, S

