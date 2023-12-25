# This is a script for training and testing the unsupervised diffusion model
#
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

def forward_process(data, T, betas):
    for t in range(T):
        beta_t = betas[t]
        mu = data * np.sqrt(1 - beta_t)
        std = np.sqrt(beta_t)
        data = mu + std * np.random.randn(data.shape[0], data.shape[1]) # data ~ N(mu, std)
    return data


def main():
    # load 2D swiss roll dataset
    n_samples = 5000
    data, _ = make_swiss_roll(n_samples)
    data = data[:, [2, 0]]/10
    data = data * np.array([1, -1])
    # plot the 2D swiss roll dataset
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.show()
    # start forward diffusion
    T = 40
    betas = np.linspace(1e-4, 1e-2, T)
    xT = forward_process(data, T, betas)
    # plot the forward diffusion result
    plt.figure(figsize=(10, 10))
    plt.scatter(xT[:, 0], xT[:, 1])
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.show()


if __name__== '__main__':
    # Define the parameters
    main()
    pass