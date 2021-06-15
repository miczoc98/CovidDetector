import tensorflow.keras as keras
import numpy as np
from lib.ModelTrainer import ModelTrainer, PreTrainedFactory


def build_and_train(path, dataset, pretrained, image_size=128, learning_rate=0.001, dropout=0.0, epochs=50,
                    fine_tuning_layers=0, fine_tuning_epochs=0):

    factory = PreTrainedFactory((image_size, image_size, 3))
    model = factory.create_pretrained(pretrained)

    trainer = ModelTrainer(model, dataset)

    trainer.build_model(dropout=dropout, learning_rate=learning_rate)
    trainer.train_model(path, epochs=epochs, fine_tuning_layers=fine_tuning_layers,
                        fine_tuning_epochs=fine_tuning_epochs)
    score = trainer.evaluate_model()
    np.savetxt(path + ".test", score, "%.3f")


if __name__ == '__main__':
    network = "vgg19"
    dataset = "datasets/dataset_300"
    ds_dir = dataset.split("/")[-1]

    image_sizes = [256, 128, 71]
    learning_rates = [0.01, 0.001, 0.0001]
    dropouts = [0.2, 0.4, 0.6]
    fine_tuning_layer_counts = [50, 100, 200]
    fine_tuning_epoch_counts = [10, 20, 50]

    for learning_rate in learning_rates:
        path = "saved_models/" + ds_dir + "/learning_rates/" + \
            network + "_" + str(learning_rate) + ".h5"
        build_and_train(path, dataset, network, learning_rate=learning_rate)

    for image_size in image_sizes:
        path = "saved_models/" + ds_dir + "/sizes/" + \
            network + "_" + str(image_size) + ".h5"
        build_and_train(path, dataset, network, image_size=image_size)

    for dropout in dropouts:
        path = "saved_models/" + ds_dir + "/dropouts/" + \
            network + "_" + str(dropout) + ".h5"
        build_and_train(path, dataset, network, dropout=dropout)

    for fine_tuning_layers in fine_tuning_layer_counts:
        path = "saved_models/" + ds_dir + "/fine_tuning_layers/" + \
            network + "_" + str(fine_tuning_layers) + ".h5"
        build_and_train(path, dataset, network,
                        fine_tuning_layers=fine_tuning_layers, fine_tuning_epochs=20)

    for fine_tuning_epochs in fine_tuning_epoch_counts:
        path = "saved_models/" + ds_dir + "/fine_tuning_epochs/" + \
            network + "_" + str(fine_tuning_epochs) + ".h5"
        build_and_train(path, dataset, network, fine_tuning_layers=100,
                        fine_tuning_epochs=fine_tuning_epochs)
