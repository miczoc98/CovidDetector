from lib.ModelTrainer import ModelTrainer, PreTrainedFactory

if __name__ == '__main__':
    factory = PreTrainedFactory((128, 128, 3))
    mobileNet = factory.create_pretrained("xception")

    trainer = ModelTrainer(mobileNet, "datasets/dataset_small")
    trainer.build_model()
    trainer.train_model("saved_models/xception.h5")
    score = trainer.evaluate_model()
    print(score)
    print("loss:" + str(score[0]))
    print("acc:" + str(score[1]))
