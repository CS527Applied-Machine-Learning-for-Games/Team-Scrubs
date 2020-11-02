import os, sys, time, PIL, threading
import numpy as np
from model_prune import unet, prune_model
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


def evaluate_thread(m, m_id):
    """
    This thread takes a model, evaluates it, and saves the result to report directory.
    ======
    Parameter:
    m - model
    m_id - model index number starting from 1
    """
    print("===Evaluating Thread for model {} Start===".format(m_id))
    print("Evaluating Thread: Loading testing dataset...")
    test_X, test_Y = test_data()
    print("Evaluating Thread: Evaluating model {}...".format(m_inm_iddex))
    results = m.evaluate(test_X, test_Y, batch_size=10)
    print("Evaluating Thread: test loss, test acc:", results)
    if not os.path.exists("../UnityProject/Assets/Resources/data/reports"):
        print("Evaluating Thread: Creating reports directory...")
        os.makedirs("../UnityProject/Assets/Resources/data/reports")
    with open("../UnityProject/Assets/Resources/data/reports/{}.txt".format(m_id), "w") as f:
        f.write(str(results[1]))
    print("Evaluating Thread: Resut saved to ../UnityProject/Assets/Resources/data/reports/{}.txt".format(m_id))
    print("===Evaluating Thread for model {} End===".format(m_id))


def save_prediction(imgs, start_index=1, end_index=801):
    if not os.path.exists("../UnityProject/Assets/Resources/data/prediction"):
        print("Creating prediction directory...")
        os.makedirs("../UnityProject/Assets/Resources/data/prediction")
    for i, img in zip(range(start_index, end_index), imgs):
        image_dir = (
            "../UnityProject/Assets/Resources/data/prediction/" + str(i) + ".png"
        )
        PIL.Image.fromarray(np.uint8(np.squeeze(img * 255))).save(image_dir)


def prediction_thread(m, m_id, X, start_index):
    print("===Predicting Thread for model {} Start===".format(m_id))
    print("Predicting game images...")
    pred_Y = m.predict(X)
    print("Saving predictions...")
    save_prediction(pred_Y)
    print("===Predicting Thread for model {} End===".format(m_id))


def pre_train():
    m = unet()
    print("Retrieving pre-train data...")
    X, Y = pretrain_data()
    H = m.fit(X, Y, batch_size=10, epochs=1, validation_split=0.8)
    print("Saving model...")
    m.save("./models/1.h5")
    print("Model saved to data/models/1.h5")
    evl_t = threading.Thread(target=evaluate_thread, args=(m, "1"), daemon=True)
    evl_t.start()
    print("Load game images...")
    X, _ = play_data()
    pred_t = threading.Thread(target=,args=,daemon=)
    pred_t.start()

    # Wait for both evaluation and prediction to finish
    evl_t.join()
    pred_t.join()
    return  m


def train_model_by_batch():
    
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
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if len(sys.argv) == 1:
        prune = False
    elif len(sys.argv) == 2:
        if sys.argv[1] == "--prune":
            prune = True
        else:
            print("Wrong argument. Currently only accept variable [--prune] .")
            exit(0)
    else:
        print("Wrong number of argument")
        exit(0)

    # Check whether there exist previous model
    if not os.path.exists("./models/"):
        print("Creating model directory...")
        os.makedirs("./models/")
        BATCH_NUM = 0
        pre_train()
    elif not os.path.exists("./models/1.h5"):
        print("Pre-training the model...")
        BATCH_NUM = 0
        pre_train()
    else:
        try:
            with open("../UnityProject/Assets/Resources/data/player_data.txt", "r") as f:
                current_img = int(f.read().strip())
                BATCH_NUM = (current_img-1)//10
                print("Player currently played {} images, the BATCH_NUM is set to {}".format(current_img, BATCH_NUM))
        except:
            print("Error when Loading player_data.txt")

    train_model_by_batch(prune)
