import os
import csv
import matplotlib.pyplot as plt
import numpy as np


def plot_init():
    image_sizes = [256, 128, 71]
    learning_rates = [0.01, 0.001, 0.0001]
    dropouts = [0.2, 0.4, 0.6]
    fine_tuning_layer_counts = [50, 100, 200]
    fine_tuning_epoch_counts = [10, 20, 50]

    data_to_plot = [dropouts, fine_tuning_epoch_counts, fine_tuning_layer_counts, learning_rates,
                    image_sizes]

    dir_names = ['dropouts', 'fine_tuning_epochs', 'fine_tuning_layers', 'learning_rates', 'sizes']
    dataset_name = 'dataset_300'

    networks = ['mobileNet', 'xception', 'vgg19']
    titles = ['Dropout', "Fine Tuning Epochs", "Fine Tuning Layers", "Learning Rate", "Image Size"]

    for tested_param, dir_name, title in zip(data_to_plot, dir_names, titles):
        acc, val_acc, loss, val_loss = read_data_from_files(dataset_name, dir_name, networks,
                                                            tested_param)
        name = f"{dir_name}_plot"

        plot_net(acc, val_acc, "acc", tested_param, networks, name, title)
        plot_net(loss, val_loss, "loss", tested_param, networks, name, title)


def read_data_from_files(dataset_name, dir_name, networks, params):
    acc = {}
    val_acc = {}
    loss = {}
    val_loss = {}

    for network in networks:
        acc[network] = []
        val_acc[network] = []
        loss[network] = []
        val_loss[network] = []

        for param in params:
            path = f"saved_models/{dataset_name}/{dir_name}/{network}_{param}.h5.history.csv"
            acc[network].append([])
            val_acc[network].append([])
            loss[network].append([])
            val_loss[network].append([])

            with open(path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 4:
                        continue
                    loss[network][-1].append(float(row[0]))
                    acc[network][-1].append(float(row[1]))
                    val_loss[network][-1].append(float(row[2]))
                    val_acc[network][-1].append(float(row[3]))

    return acc, val_acc, loss, val_loss


def plot_net(p1, p2, param_name, params, networks, name, title):
    def find_range():
        min_val = np.inf
        max_val = np.NINF
        for p in (p1, p2):
            for arr_arr in list(p.values()):
                for arr in arr_arr:
                    min_val = min(min(arr), min_val)
                    max_val = max(max(arr), max_val)

        return min_val - 0.05, max_val + 0.05

    fig, axes = plt.subplots(3, 1, figsize=(15, 11))

    fig.suptitle(title, x=0.01, ha="left", fontsize="xx-large")

    axes[0].set_title('Xception')
    axes[1].set_title('MobileNet')
    axes[2].set_title('VGG19')

    for ax in axes:
        ax.set_xlabel('epochs')
        ax.set_ylabel(param_name)
        ax.set_ylim(find_range())
        ax.margins(0.01, 0)

    for ax, network in zip(axes, networks):
        colors = ['#D3212D', '#8DB600', '#007FFF']
        for train, val, param, c in zip(p1[network], p2[network], params, colors):
            ax.plot(train, label=str(param), color=c)
            ax.plot(val, linestyle=":", color=c)
        if param_name == "acc":
            ax.legend(loc='lower right')
        else:
            ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f"./plots/{param_name}_{name}.png")
    plt.close()


if __name__ == '__main__':
    plot_init()
    plt.savefig("plot.png")
