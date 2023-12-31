# This is a script for training and testing the unsupervised diffusion model
#
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
import torch.nn as nn

def forward_process(data, T, betas):
    for t in range(T):
        beta_t = betas[t]
        mu = data * torch.sqrt(1 - beta_t)
        std = torch.sqrt(beta_t)
        # Sample from q(x_t | x_{t-1})
        data = mu + std * torch.randn_like(data) # data ~ N(mu, std)
    return data

class MLP(nn.Module):
    def __init__(self, N=40, data_dim=2, hidden_dim=64):
        super(MLP, self).__init__()
        self.network_head = nn.Sequential(nn.Linear(data_dim, hidden_dim), \
                                          nn.ReLU(), \
                                          nn.Linear(hidden_dim, hidden_dim), \
                                          nn.ReLU(), )
        self.network_tail = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), \
                                                            nn.ReLU(), \
                                                            nn.Linear(hidden_dim, data_dim * 2),) for t in range(N)])
    def forward(self, x, t):
        h = self.network_head(x) # [batch_size, hidden_dim]
        tmp = self.network_tail[t](h) # [batch_size, data_dim * 2]
        mu, h = torch.chunk(tmp, 2, dim=1)
        var = torch.exp(h)
        std = torch.sqrt(var)
        return mu, std

class DiffusionModel():
    # an efficient implementation of the forward diffusion process
    def __init__(self, T, model: nn.Module, dim=2):
        self.T = T
        self.model = model
        self.dim = dim
        self.betas = torch.sigmoid(torch.linspace(-18, 10, T)) * (3e-1 - 1e-5) + 1e-5
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        print('alpha = ' + str(self.alphas))
        print('alpha_bar = ' + str(self.alphas_bar))
        print('The diffusion step T is ' + str(T))

    def forward_process(self, x0, t):
        """
        :param t: number of diffusion steps
        """
        assert t > 0,  "t must be greater than 0"
        assert t <= self.T,  f"t must be less than {self.T}"
        print('The diffusion step t is ' + str(t))
        t = t - 1
        mu = torch.sqrt(self.alphas_bar[t]) * x0
        std= torch.sqrt(1 - self.alphas_bar[t])
        epsilon = torch.randn_like(x0)
        xt = mu + std * epsilon # data ~ N(mu, std)

        std_q = torch.sqrt((1 - self.alphas_bar[t-1])/ (1 - self.alphas_bar[t]) * self.betas[t])
        m1 = torch.sqrt(self.alphas_bar[t - 1]) * self.betas[t] / (1 - self.alphas_bar[t])
        m2 = torch.sqrt(self.alphas[t]) * (1 - self.alphas_bar[t - 1]) / (1 - self.alphas_bar[t])
        mu_q = m1 * x0 + m2 * xt
        return mu_q, std_q, xt

    def reverse_process(self, xT, t):
        """
        :param t: number of diffusion steps
        """
        assert t > 0,  "t must be greater than 0"
        assert t <= self.T,  f"t must be less than {self.T}"
        print('The diffusion step t is ' + str(t))
        t = t - 1
        mu, std = self.model(xT, t)
        epsilon = torch.randn_like(xT)
        return mu + std * epsilon # x0 ~ N(mu, std)

    def sample(self, batch_size, device):
        noise = torch.randn((batch_size, self.dim)).to(device)
        x = noise
        samples = [x]
        for t in range(self.T, 0, -1):
            if not (t == 1):
                x = self.reverse_process(x, t)
            samples.append(x)
        return samples[::-1]

    def get_loss(self, x0):
        """
        :param x0: batch[batch_size, self.dim]
        """
        t = torch.randint(low=2, high=40+1, size=(1,))
        mu_q, sigma_q, xt = self.forward_process(x0, t)
        mu_p, sigma_p, xt_minus1 = self.reverse_process(xt.float(), t)
        KL = torch.log(sigma_p) - torch.log(sigma_q) + (
                sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2 * sigma_p ** 2)
        K = - KL.mean()  # Should be maximized
        loss = - K  # Should be minimized
        return loss

def main():
    # load the swiss roll dataset
    n_samples = 10_000
    data, _ = make_swiss_roll(n_samples)
    data = data[:, [2, 0]]/10
    data = data * np.array([1, -1])
    # plot the swiss roll dataset
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.show()

    T = 40
    t = 40
    # start forward diffusion
    #betas = torch.sigmoid(torch.linspace(-18, 10, T)) * (3e-1 - 1e-5) + 1e-5
    #xT = forward_process(torch.from_numpy(data), T, betas)
    # test the efficient forward process
    model = DiffusionModel(T)
    x0 = torch.from_numpy(data)
    xT = model.forward_process(x0, t)
    # plot the forward diffusion result
    print(xT.mean(0))
    print(xT.std(0))
    plt.figure(figsize=(10, 10))
    plt.scatter(xT[:, 0].data.numpy(), xT[:, 1].data.numpy(), alpha=0.1)
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.show()

    # reverse process
    x0 = model.reverse_process(xT, t)






if __name__== '__main__':
    # Define the parameters
    main()
    pass