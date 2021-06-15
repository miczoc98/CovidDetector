test_params = ["fine_tuning_epochs", "fine_tuning_layers"]
networks = ["mobileNet", "xception", "vgg19"]
params = [["10", "20", "50"], ["50", "100", "200"]]

for i in range(len(test_params)):
    for param in params[i]:
        for network in networks:
            with open(f"saved_models/dataset_300/{test_params[i]}/{network}_{param}.h5.history.csv", "a") as f:
                with open(f"saved_models/dataset_300/{test_params[i]}/{network}_{param}.h5.fine_tuning.history.csv") as f2:
                    f.writelines(f2)
