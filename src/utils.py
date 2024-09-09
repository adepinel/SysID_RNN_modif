import torch


def calculate_collisions(x, sys, min_dist):
    deltax = x[:, 0::4].repeat(sys.n_agents, 1, 1) - x[:, 0::4].repeat(sys.n_agents, 1, 1).transpose(0, 2)
    deltay = x[:, 1::4].repeat(sys.n_agents, 1, 1) - x[:, 1::4].repeat(sys.n_agents, 1, 1).transpose(0, 2)
    distance_sq = (deltax ** 2 + deltay ** 2)
    n_coll = ((0.0001 < distance_sq) * (distance_sq < min_dist**2)).sum()
    return n_coll

def set_params():
    # # # # # # # # Parameters # # # # # # # #
    min_dist = 1.  # min distance for collision avoidance
    t_end = 5#100
    n_agents = 2
    x0, xbar = get_ini_cond(n_agents)
    linear = False
    # # # # # # # # Hyperparameters # # # # # # # #
    learning_rate = 1e-3
    epochs = 2#500
    Q = torch.kron(torch.eye(n_agents), torch.diag(torch.tensor([1, 1, 1, 1.])))
    alpha_u = 0.1  # Regularization parameter for penalizing the input
    alpha_ca = 100
    alpha_obst = 5e3
    n_xi = 32  # \xi dimension -- number of states of REN
    l = 32  # dimension of the square matrix D11 -- number of _non-linear layers_ of the REN
    n_traj = 1#20  # number of trajectories collected at each step of the learning
    std_ini = 0.2  # standard deviation of initial conditions
    gamma_bar = 100
    return min_dist, t_end, n_agents, x0, xbar, linear, learning_rate, epochs, Q, alpha_u, alpha_ca, alpha_obst, n_xi, \
           l, n_traj, std_ini, gamma_bar


def get_ini_cond(n_agents):
    # Corridor problem
    x0 = torch.tensor([2., -2, 0, 0,
                       -2, -2, 0, 0,
                       ])
    xbar = torch.tensor([-2, 2, 0, 0,
                         2., 2, 0, 0,
                         ])
    return x0, xbar
