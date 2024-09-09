#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class PsiX(nn.Module):
    def __init__(self, f):
        super().__init__()
        n = 4
        m = 2
        self.f = f

    def forward(self, t, omega):
        y, u = omega
        psi_x = self.f(t, y, u)
        omega_ = 0
        return psi_x, omega_


class Controller(nn.Module):
    def __init__(self, f, n, m, n_xi, l, gammabar):
        super().__init__()
        self.n = n
        self.m = m
        self.psi_x = PsiX(f)
        self.psi_u = RenG(self.n, self.m, n_xi, l, bias=False, mode="l2stable", gamma=gammabar)

    def forward(self, t, y_, xi, omega):
        psi_x, _ = self.psi_x(t, omega)
        w_ = y_ - psi_x
        u_, xi_ = self.psi_u(t, w_, xi)
        omega_ = (y_, u_)
        return u_, xi_, omega_


class TwoRobots(torch.nn.Module):
    def __init__(self, xbar, linear=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.xbar = xbar
        self.n_agents = 2
        self.n = 8
        self.m = 4
        self.h = 0.05

        self.Mx = torch.zeros(self.n_agents, self.n, device=device)
        self.Mx[0, 0] = 1
        self.Mx[1, 4] = 1
        self.My = torch.zeros(self.n_agents, self.n, device=device)
        self.My[0, 1] = 1
        self.My[1, 5] = 1

        self.Mvx = torch.zeros(self.n_agents, self.n, device=device)
        self.Mvx[0, 2] = 1
        self.Mvx[1, 6] = 1
        self.Mvy = torch.zeros(self.n_agents, self.n, device=device)
        self.Mvy[0, 3] = 1
        self.Mvy[1, 7] = 1

        self.mv1 = torch.zeros(2, self.n, device=device)
        self.mv1[0, 2] = 1
        self.mv1[1, 3] = 1
        self.mv2 = torch.zeros(2, self.n, device=device)
        self.mv2[0, 6] = 1
        self.mv2[1, 7] = 1
        self.mp = torch.zeros(2, self.n, device=device)
        self.Mp = torch.cat((self.mv1, self.mp, self.mv2, self.mp), 0)

    def f(self, t, x, u, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        m1, m2 = 1, 1
        kspringGround = 2
        cdampGround = 2


        k1, k2 = kspringGround, kspringGround
        c1, c2 = cdampGround, cdampGround

        px = torch.matmul(self.Mx, x)
        py = torch.matmul(self.My, x)
        vx = torch.matmul(self.Mvx, x)
        vy = torch.matmul(self.Mvy, x)



        B = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1 / m1, 0, 0, 0],
            [0, 1 / m1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1 / m2, 0],
            [0, 0, 0, 1 / m2],
        ])

        xt = torch.matmul(self.Mx.float(), self.xbar.float())
        yt = torch.matmul(self.My, self.xbar.float())

        deltaxt = px - xt
        deltayt = py - yt

        projxt = torch.cos(torch.atan2(deltayt, deltaxt))
        projyt = torch.sin(torch.atan2(deltayt, deltaxt))
        projvxt = torch.cos(torch.atan2(vy, vx))
        projvyt = torch.sin(torch.atan2(vy, vx))

        Fc01 = c1 * torch.sqrt(vx[0] ** 2 + vy[0] ** 2)
        Fc02 = c2 * torch.sqrt(vx[1] ** 2 + vy[1] ** 2)

        Fk01 = k1 * torch.sqrt(deltaxt[0] ** 2 + deltayt[0] ** 2)
        Fk02 = k2 * torch.sqrt(deltaxt[1] ** 2 + deltayt[1] ** 2)


        Fground1x = -Fk01 * projxt[0] - Fc01 * projvxt[0]
        Fground1y = -Fk01 * projyt[0] - Fc01 * projvyt[0]

        Fground2x = -Fk02 * projxt[1] - Fc02 * projvxt[1]
        Fground2y = -Fk02 * projyt[1] - Fc02 * projvyt[1]


        A1x = torch.tensor([
            0,
            0,
            (Fground1x) / m1,
            (Fground1y) / m1,
            0,
            0,
            (Fground2x) / m2,
            (Fground2y) / m2,
        ])

        A2x = torch.matmul(self.Mp, x)

        Ax = A1x + A2x

        x1 = x + (Ax + torch.matmul(B, u)) * self.h
        return x1

    def f_withnoise(self, t, x, u, w, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        #matrix used to apply the noise only to the position and scaled with time
        Mw = torch.zeros(self.n, self.n, device=device)
        Mw[0, 0] = 15 / (t + 1)
        Mw[1, 1] = 15 / (t + 1)
        Mw[4, 4] = 15 / (t + 1)
        Mw[5, 5] = 15 / (t + 1)
        x1 = self.f(t, x, u, w)+torch.matmul(Mw, w) * self.h
        return x1

    def forward(self, t, x, u, w):
        if t == 0:
            x1 = w
        else:
            x1 = self.f_withnoise(t, x, u, w)
        return x1, x1

class SystemRobots(nn.Module):
    def __init__(self, xbar, linear=True):
        super().__init__()
        self.n_agents = int(xbar.shape[0]/4)
        self.n = 4*self.n_agents
        self.m = 2*self.n_agents
        self.h = 0.05
        self.mass = 1.0
        self.k = 1.0
        self.b = 1.0
        if linear:
            self.b2 = 0
        else:
            self.b2 = 0.1
        m = self.mass
        self.B = torch.kron(torch.eye(self.n_agents),
                            torch.tensor([[0, 0],
                                          [0., 0],
                                          [1/m, 0],
                                          [0, 1/m]])
                            )*self.h
        self.xbar = xbar

    def A(self, x):
        b2 = self.b2
        b1 = self.b
        m, k = self.mass, self.k
        A1 = torch.eye(4*self.n_agents)
        A2 = torch.cat((torch.cat((torch.zeros(2,2),
                                   torch.eye(2)
                                   ), dim=1),
                        torch.cat((torch.diag(torch.tensor([-k/m, -k/m])),
                                   torch.diag(torch.tensor([-b1/m, -b1/m]))
                                   ),dim=1),
                        ),dim=0)
        A2 = torch.kron(torch.eye(self.n_agents), A2)
        mask = torch.tensor([[0, 0], [1, 1]]).repeat(self.n_agents, 1)
        A3 = torch.norm(x.view(2 * self.n_agents, 2) * mask, dim=1, keepdim=True)
        A3 = torch.kron(A3, torch.ones(2,1))
        A3 = -b2/m * torch.diag(A3.squeeze())
        A = A1 + self.h * (A2 + A3)
        return A

    def f(self, t, x, u):
        sat = False
        if sat:
            v = torch.ones(self.m)
            u = torch.minimum(torch.maximum(u, -v), v)
        f = F.linear(x - self.xbar, self.A(x)) + F.linear(u, self.B) + self.xbar
        return f

    def forward(self, t, x, u, w):
        Mw = torch.zeros(8, 8)
        Mw[0, 0] = 15 / (t + 1)
        Mw[1, 1] = 15 / (t + 1)
        Mw[4, 4] = 15 / (t + 1)
        Mw[5, 5] = 15 / (t + 1)
        x_ = self.f(t, x, u) + torch.matmul(Mw, w) * self.h  # here we can add noise not modelled
        y = x_
        return x_, y
