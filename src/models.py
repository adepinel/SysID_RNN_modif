#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# REN implementation in the acyclic version
# See paper: "Recurrent Equilibrium Networks: Flexible dynamic models with guaranteed stability and robustness"

class PsiU(nn.Module):
    def __init__(self, n, m, n_xi, l):
        super().__init__()
        self.n = n  # nel paper m
        self.n_xi = n_xi  # nel paper n1
        self.l = l  # nel paper q
        self.m = m  # nel paper p1
        self.s = np.max((n, m))  # s nel paper, dimensione di X3 Y3

        # # # # # # # # # Training parameters # # # # # # # # #
        # Auxiliary matrices:
        std = 1
        self.X = nn.Parameter((torch.randn(2 * n_xi + l, 2 * n_xi + l) * std))
        self.Y = nn.Parameter((torch.randn(n_xi, n_xi) * std))  # Y1 nel paper
        # NN state dynamics:
        self.B2 = nn.Parameter((torch.randn(n_xi, n) * std))
        # NN output:
        self.C2 = nn.Parameter((torch.randn(m, n_xi) * std))
        self.D21 = nn.Parameter((torch.randn(m, l) * std))
        self.X3 = nn.Parameter(torch.randn(self.s, self.s) * std)
        self.Y3 = nn.Parameter(torch.randn(self.s, self.s) * std)
        self.sg = nn.Parameter(torch.randn(1, 1) * std)  # square root of gamma

        # v signal:
        self.D12 = nn.Parameter((torch.randn(l, n) * std))
        # bias:
        # self.bxi = nn.Parameter(torch.randn(n_xi))
        # self.bv = nn.Parameter(torch.randn(l))
        # self.bu = nn.Parameter(torch.randn(m))
        # # # # # # # # # Non-trainable parameters # # # # # # # # #
        # Auxiliary elements
        self.epsilon = 0.001
        self.F = torch.zeros(n_xi, n_xi)
        self.B1 = torch.zeros(n_xi, l)
        self.E = torch.zeros(n_xi, n_xi)
        self.Lambda = torch.ones(l)
        self.C1 = torch.zeros(l, n_xi)
        self.D11 = torch.zeros(l, l)
        self.Lq = torch.zeros(m, m)
        self.Lr = torch.zeros(n, n)
        self.D22 = torch.zeros(m, n)
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
        v = F.linear(xi, self.C1[0, :]) + F.linear(w,
                                                   self.D12[0, :])  # + self.bv[0]
        epsilon = epsilon + vec * torch.tanh(v / self.Lambda[0])
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


class PsiU2(nn.Module):
    def __init__(self, n, m, n_xi, l):
        super().__init__()
        self.n = n
        self.n_xi = n_xi
        self.l = l
        self.m = m
        # # # # # # # # # Training parameters # # # # # # # # #
        # Auxiliary matrices:
        std = 0.1
        self.X = nn.Parameter((torch.randn(2*n_xi+l, 2*n_xi+l)*std))
        self.Y = nn.Parameter((torch.randn(n_xi, n_xi)*std))
        # NN state dynamics:
        self.B2 = nn.Parameter((torch.randn(n_xi, n)*std))
        # NN output:
        self.C2 = nn.Parameter((torch.randn(m, n_xi)*std))
        self.D21 = nn.Parameter((torch.randn(m, l)*std))
        self.D22 = nn.Parameter((torch.randn(m, n)*std))
        # v signal:
        self.D12 = nn.Parameter((torch.randn(l, n)*std))
        # bias:
        # self.bxi = nn.Parameter(torch.randn(n_xi))
        # self.bv = nn.Parameter(torch.randn(l))
        # self.bu = nn.Parameter(torch.randn(m))
        # # # # # # # # # Non-trainable parameters # # # # # # # # #
        # Auxiliary elements
        self.epsilon = 0.001
        self.F = torch.zeros(n_xi, n_xi)
        self.B1 = torch.zeros(n_xi, l)
        self.E = torch.zeros(n_xi, n_xi)
        self.Lambda = torch.ones(l)
        self.C1 = torch.zeros(l, n_xi)
        self.D11 = torch.zeros(l, l)
        self.set_model_param()

    def set_model_param(self):
        n_xi = self.n_xi
        l = self.l
        H = torch.matmul(self.X.T, self.X) + self.epsilon * torch.eye(2*n_xi+l)
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
        v = F.linear(xi, self.C1[0,:]) + F.linear(w, self.D12[0,:])  # + self.bv[0]
        epsilon = epsilon + vec * torch.tanh(v/self.Lambda[0])
        for i in range(1, self.l):
            vec = torch.zeros(self.l)
            vec[i] = 1
            v = F.linear(xi, self.C1[i,:]) + F.linear(epsilon, self.D11[i,:]) + F.linear(w, self.D12[i,:])  # self.bv[i]
            epsilon = epsilon + vec * torch.tanh(v/self.Lambda[i])
        E_xi_ = F.linear(xi, self.F) + F.linear(epsilon, self.B1) + F.linear(w, self.B2)  # + self.bxi
        xi_ = F.linear(E_xi_, self.E.inverse())
        u = F.linear(xi, self.C2) + F.linear(epsilon, self.D21) + F.linear(w, self.D22)  # + self.bu
        return u, xi_


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
    def __init__(self, f, n, m, n_xi, l):
        super().__init__()
        self.n = n
        self.m = m
        self.psi_x = PsiX(f)
        self.psi_u = PsiU(self.n, self.m, n_xi, l)

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
