import os
import csv
import matplotlib.pyplot as plt
import numpy as np


def plot_net(param1, param2, name, title):
    plt.figure(figsize=(11, 5))
    plt.title(title, size="xx-large")
    plt.plot(param1, label="train")
    plt.plot(param2, label="validate")
    plt.xlabel('epochs', fontsize="x-large")
    plt.ylabel(name, fontsize="x-large")
    plt.ylim(round(np.min(param1 + param2), 1) - 0.05,
             round(np.max(param1 + param2), 1) + 0.05)
    plt.margins(0.01, 0)
    plt.xticks(fontsize="large")
    plt.yticks(fontsize="large")
    if name == "acc":
        plt.legend(loc="lower right", fontsize="x-large")
    else:
        plt.legend(loc="upper right", fontsize="x-large")
    plt.savefig(f"final_{name}.png")
    plt.close()


if __name__ == '__main__':
    acc = []
    val_acc = []
    loss = []
    val_loss = []

    with open("final_history.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            loss.append(float(row[0]))
            acc.append(float(row[1]))
            val_loss.append(float(row[2]))
            val_acc.append(float(row[3]))

    plot_net(acc, val_acc, "acc", "MobileNet")
    plot_net(loss, val_loss, "loss", "MobileNet")


