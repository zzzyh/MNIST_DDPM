# This is a script for training and testing the DDPM model on MNIST dataset.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
def main():
    # load 2D swiss roll dataset
    n_samples = 5000
    data, _ = make_swiss_roll(n_samples)
    data = data[:, [2, 0]]/10
    data = data * np.array([1, -1])
    print(data)
    # plot the 2D swiss roll dataset
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.show()



if __name__== '__main__':
    # Define the parameters
    main()
    pass