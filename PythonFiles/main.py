import os, sys, time, PIL
import numpy as np
from model import unet
from dataLoader import (
    pretrain_data,
    test_data,
    play_data,
    play_data_by_batch,
    get_data_x,
    play_data_by_batch_x,
)

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler


def save_prediction(imgs, start_index=1, end_index=801):
    if not os.path.exists("../UnityProject/Assets/Resources/data/prediction"):
        print("Creating prediction directory...")
        os.makedirs("../UnityProject/Assets/Resources/data/prediction")
    for i, img in zip(range(start_index, end_index), imgs):
        image_dir = (
            "../UnityProject/Assets/Resources/data/prediction/" + str(i) + ".png"
        )
        PIL.Image.fromarray(np.uint8(np.squeeze(img * 255))).save(image_dir)


def pre_train():
    m = unet()
    print("Retrieving pre-train data...")
    X, Y = pretrain_data()
    H = m.fit(X, Y, batch_size=10, epochs=1, validation_split=0.8)
    print("Saving model...")
    m.save("./models/1.h5")
    print("Model saved to data/models/1.h5")
    print("Evaluating pre-trained model...")
    results = m.evaluate(test_X, test_Y, batch_size=10)
    print("test loss, test acc:", results)
    if not os.path.exists("../UnityProject/Assets/Resources/data/reports"):
        print("Creating reports directory...")
        os.makedirs("../UnityProject/Assets/Resources/data/reports")
    with open("../UnityProject/Assets/Resources/data/reports/1.txt", "w") as f:
        f.write(str(results[1]))
    print("Resut saved to data/reports/1.txt")
    print("Load game images...")
    X, _ = play_data()
    print("Predicting game images...")
    pred_Y = m.predict(X)
    print("Saving predictions...")
    save_prediction(pred_Y)


def train_model_by_batch():
    BATCH_NUM = 0
    # Check if pre-trained model exist
    path_to_watch = "../UnityProject/Assets/Resources/data/drawings/"
    before = dict(
        [(f, None) for f in os.listdir(path_to_watch) if f.split(".")[-1] == "png"]
    )
    while True:
        time.sleep(10)
        after = dict(
            [(f, None) for f in os.listdir(path_to_watch) if f.split(".")[-1] == "png"]
        )
        added = [f for f in after if not f in before]
        if added:
            print("Added drawings:", ", ".join(added))
        before = after
        if len(before) >= BATCH_NUM * 10 + 10:
            # Train model for 1 batch
            X, Y = play_data_by_batch(BATCH_NUM)
            Y = np.array([y[:, :, :, 0] for y in Y])
            m = load_model("./models/{}.h5".format(BATCH_NUM + 1))
            m.fit(X, Y, batch_size=10, epochs=1)
            BATCH_NUM += 1
            m.save("./models/{}.h5".format(BATCH_NUM + 1))
            print("Model saved to models/{}.h5".format(BATCH_NUM + 1))
            ## How to evaluate?     **Could add metric to model.py or design our own
            print("Evaluating new model...")
            results = m.evaluate(test_X, test_Y, batch_size=10)
            print("test loss, test acc:", results)
            with open(
                "../UnityProject/Assets/Resources/data/reports/{}.txt".format(
                    BATCH_NUM + 1
                ),
                "w",
            ) as f:
                f.write(str(results[1]))
            print("Result saved to data/reports/{}.txt".format(BATCH_NUM + 1))

            next_X = play_data_by_batch_x(BATCH_NUM)
            pred_Y = m.predict(next_X)
            save_prediction(pred_Y, BATCH_NUM * 10 + 1, BATCH_NUM * 10 + 11)
        else:
            print("currently detect ", len(before), ", waiting for more drawing ...")


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    if not os.path.exists("./models/"):
        print("Creating model directory...")
        os.makedirs("./models/")
    print("Preparing testing data...")
    test_X, test_Y = test_data()
    if not os.path.exists("./models/1.h5"):
        print("Pre-training the model...")
        pre_train()
    train_model_by_batch()
