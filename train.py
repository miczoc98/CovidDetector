from lib.ModelTrainer import ModelTrainer, PreTrainedFactory

if __name__ == '__main__':
    factory = PreTrainedFactory((128, 128, 3))
    mobileNet = factory.create_pretrained("mobileNet")

    trainer = ModelTrainer(mobileNet, "dataset")
    trainer.build_model()
    trainer.train_model("saved_models/mobileNet.h5")
    score = trainer.evaluate_model()
    print("loss:" + score[0])
    print("acc:" + score[1])
