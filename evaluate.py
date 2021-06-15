import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow.keras as keras

if __name__ == '__main__':
    model = keras.models.load_model("saved_models/vgg19_50.h5")

    generator = keras.preprocessing.image.ImageDataGenerator()

    validation_gen = generator.flow_from_directory(directory="datasets/dataset_3000/test",
                                                   batch_size=32,
                                                   target_size=(128, 128),
                                                   shuffle=False,
                                                   class_mode="sparse")

    y_pred = model.predict_generator(validation_gen)
    y_pred = np.argmax(y_pred, axis=1)

    target_names = ["covid", "normal", 'pneumonia']

    print(confusion_matrix(validation_gen.classes, y_pred))
    print(classification_report(validation_gen.classes, y_pred, target_names=target_names))