import tensorflow.keras as keras

from lib.ModelTrainer import ModelTrainer, PreTrainedFactory

if __name__ == '__main__':
    factory = PreTrainedFactory((256, 256, 3))
    model = factory.create_pretrained("xception")

    trainer = ModelTrainer(model, "datasets/dataset_3000")

    trainer.build_model(dropout=0.02, learning_rate=0.001)
    trainer.train_model("saved_models/xception.h5", epochs=10, fine_tuning_layers=100, fine_tuning_epochs=10)
    score = trainer.evaluate_model()
    print("loss:" + str(score[0]))
    print("acc:" + str(score[1]))
