import csv
import os
from typing import Tuple

import numpy as np
import tensorflow.keras as keras
from keras.preprocessing.image_dataset import image_dataset_from_directory


class PreTrainedFactory:
    def __init__(self, input_size: Tuple[int, int, int]):
        self.input_size = input_size

    def create_pretrained(self, net: str) -> keras.Model:
        if net == "mobileNet":
            return self._create_mobile_net()
        if net == "xception":
            return self._create_xception()
        if net == "vgg19":
            return self._create_vgg19()
        else:
            raise Exception("no " + net + " network preset")

    def _create_mobile_net(self) -> keras.Model:
        model_base = keras.applications.MobileNetV2(input_shape=self.input_size, include_top=False, weights='imagenet')
        model_base.trainable = False

        preprocess_input_layer = keras.applications.mobilenet_v2.preprocess_input

        return self._create_model(model_base, preprocess_input_layer)

    def _create_xception(self) -> keras.Model:
        model_base = keras.applications.Xception(input_shape=self.input_size, include_top=False, weights='imagenet')
        model_base.trainable = False

        preprocess_input_layer = keras.applications.xception.preprocess_input
        return self._create_model(model_base, preprocess_input_layer)

    def _create_vgg19(self) -> keras.Model:
        model_base = keras.applications.MobileNetV2(input_shape=self.input_size, include_top=False, weights='imagenet')
        model_base.trainable = False

        preprocess_input_layer = keras.applications.vgg19.preprocess_input
        return self._create_model(model_base, preprocess_input_layer)

    def _create_model(self, model_base, preprocess_input_layer):
        inputs = keras.Input(shape=self.input_size)
        x = preprocess_input_layer(inputs)
        outputs = model_base(x, training=False)
        model = keras.Model(inputs, outputs)
        return model


class ModelTrainer:
    def __init__(self, base_model: keras.Model, dataset_path: str = "dataset"):
        self.base_model = base_model
        self.dataset_path = dataset_path

        self.model = None
        self.training_history = None

        self.classes = self._create_class_list()
        self.input_size = self._get_input_size()
        self.output_size = len(self.classes)

    def build_model(self):
        output_layers = [
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(self.output_size)
        ]

        model = keras.Sequential([self.base_model, *output_layers])

        model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(
            from_logits=True), metrics=['accuracy'])

        self.model = model

    def train_model(self, path: str, epochs: int = 10):
        batch_size = 32

        train_image_dir = self.dataset_path + "/train"
        validation_image_dir = self.dataset_path + "/validate"

        train_data_gen = self._create_train_data_generator(batch_size, train_image_dir)
        val_data_gen = self._create_validation_data_generator(batch_size, validation_image_dir)

        training = self.model.fit(train_data_gen, epochs=epochs, validation_data=val_data_gen)
        self._save_history(path, training.history)

        self.model.save(path)
        self._save_labels(path + ".labels.csv")

    def evaluate_model(self) -> list:
        generator = keras.preprocessing.image.ImageDataGenerator()

        images = generator.flow_from_directory(self.dataset_path + "/test",
                                               batch_size=32,
                                               shuffle=True,
                                               target_size=self.input_size,
                                               classes=self.classes,
                                               class_mode="sparse")

        return self.model.evaluate(images)

    def _get_input_size(self) -> Tuple[int, int]:
        return self.base_model.input_shape[1:3]

    def _create_class_list(self) -> list:
        return os.listdir(self.dataset_path + "/train")

    def _create_train_data_generator(self, batch_size: int, directory: str):
        generator = keras.preprocessing.image.ImageDataGenerator(rotation_range=45,
                                                                 width_shift_range=.15,
                                                                 height_shift_range=.15,
                                                                 horizontal_flip=True,
                                                                 zoom_range=0.5
                                                                 )

        return generator.flow_from_directory(directory=directory,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             target_size=self.input_size,
                                             classes=self.classes,
                                             class_mode="sparse")

    def _create_validation_data_generator(self, batch_size: int, directory: str):
        generator = keras.preprocessing.image.ImageDataGenerator()

        return generator.flow_from_directory(directory=directory,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             target_size=self.input_size,
                                             classes=self.classes,
                                             class_mode="sparse")

    def _save_history(self, path, history):
        header = ["loss, acc, test_loss, test_acc"]
        values = np.transpose(list(history.values()))
        with open(path + ".history.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(values)

    def _save_labels(self, path):
        with open(path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(self.classes)
