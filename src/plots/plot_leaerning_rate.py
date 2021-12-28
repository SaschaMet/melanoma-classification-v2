import matplotlib.pyplot as plt


def plot_lr(scheduler, epochs):
    """
    Plots the learning rate for each epoch
    """
    rates = []
    for epoch in range(0, epochs):
        x = scheduler(epoch)
        rates.append(x)

    plt.xlabel('Iterations (epochs)')
    plt.ylabel('Learning rate')
    plt.plot(range(epochs), rates)
