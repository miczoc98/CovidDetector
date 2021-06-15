import os
import csv
import matplotlib.pyplot as plt
import numpy as np



def plot_init():
    image_sizes = [256, 128, 71]
    learning_rates = [0.01, 0.001, 0.0001]
    dropouts = [0.2, 0.4, 0.6]
    epoch_counts = [10, 20, 50]
    fine_tuning_layer_counts = [50, 100, 200]
    fine_tuning_epoch_counts = [10, 20, 50]

    data_to_plot = [dropouts, epoch_counts, fine_tuning_epoch_counts, fine_tuning_layer_counts, learning_rates,
                    image_sizes]

    dir_names = ['dropouts', 'epochs', 'fine_tuning_epochs', 'fine_tuning_layers', 'learning_rates', 'sizes']
    dataset_name = 'dataset_300'

    networks = ['mobileNet', 'xception', 'vgg19']

    for i in range(len(data_to_plot)):
        for j in range(len(image_sizes)):
            acc, val_acc, loss, val_loss = read_data_from_files(dataset_name, dir_names[i], networks, data_to_plot[i][j])
            name = f"{dir_names[i]}_{data_to_plot[i][j]}_plot"

            plot_acc(acc, val_acc, name)
            plot_loss(loss, val_loss, name)


def read_data_from_files(dataset_name, dir_name, networks, param, fine_tune: False):
    acc = []
    val_acc = []
    loss = []
    val_loss = []

    for network in networks:

        path = f"saved_models/{dataset_name}/{dir_name}/{network}_{param}.h5.history.csv"

        acc.append([])
        val_acc.append([])
        loss.append([])
        val_loss.append([])

        with open(path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 4:
                    continue
                loss[-1].append(float(row[0]))
                acc[-1].append(float(row[1]))
                val_loss[-1].append(float(row[2]))
                val_acc[-1].append(float(row[3]))

    return acc, val_acc, loss, val_loss


def plot_acc(acc, val_acc, name):
    fig, axes = plt.subplots(1, 3)

    axes[0].set_title('Xception')
    axes[1].set_title('MobileNet')
    axes[2].set_title('VGG19')

    for ax in axes:
        ax.set_xlabel('epochs')
        ax.set_ylabel('acc')
        ax.set_ylim(round(np.min(acc + val_acc), 1) - 0.05, round(np.max(acc + val_acc), 1) + 0.05)

    for i in range(len(axes)):
        axes[i].plot(acc[i], label="train")
        axes[i].plot(val_acc[i], label="validation")
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(f"./plots/acc_{name}.png")
    plt.close()


def plot_loss(loss, val_loss, name):
    fig, axes = plt.subplots(1, 3)

    axes[0].set_title('Xception')
    axes[1].set_title('MobileNet')
    axes[2].set_title('VGG19')

    for ax in axes:
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')
        ax.set_ylim(round(np.min(loss + val_loss), 1) - 0.05, round(np.max(loss + val_loss), 1) + 0.05)

    for i in range(len(axes)):
        axes[i].plot(loss[i], label="train")
        axes[i].plot(val_loss[i], label="validation")
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(f"./plots/loss_{name}.png")
    plt.close()


if __name__ == '__main__':
    plot_init()

    plt.savefig("plot.png")
