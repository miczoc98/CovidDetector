import os
import csv
import matplotlib.pyplot as plt
import numpy as np

def plot_net(param1, param2, name):
    plt.plot(param1, label="train")
    plt.plot(param2, label="validate")
    plt.xlabel('epochs')
    plt.ylabel(name)
    plt.ylim(round(np.min(param1 + param2), 1) - 0.05, round(np.max(param1 + param2), 1) + 0.05)

    plt.savefig(f"final_{name}.png")
    plt.close()

if __name__ == '__main__':
    acc = []
    val_acc = []
    loss = []
    val_loss = []

    with open("final_history.csv") as f:
        acc.append([])
        val_acc.append([])
        loss.append([])
        val_loss.append([])

        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            loss[-1].append(float(row[0]))
            acc[-1].append(float(row[1]))
            val_loss[-1].append(float(row[2]))
            val_acc[-1].append(float(row[3]))

    plot_net(acc, val_acc, "acc")
    plot_net(loss, val_loss, "loss")


