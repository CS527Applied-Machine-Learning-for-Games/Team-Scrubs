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
from util import (
    load_state, 
    load_config, 
    write_config, 
    write_state
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
    if not os.path.exists(DATA_DIR+"reports"):
        print("Evaluating Thread: Creating reports directory...")
        os.makedirs("../UnityProject/Assets/Resources/data/reports")
    with open("../UnityProject/Assets/Resources/data/reports/{}.txt".format(m_id), "w") as f:
        f.write(str(results[1]))
    print("Evaluating Thread: Resut saved to ../UnityProject/Assets/Resources/data/reports/{}.txt".format(m_id))
    print("===Evaluating Thread for model {} End===".format(m_id))


def save_prediction(imgs, start_index, end_index):
    if not os.path.exists("../UnityProject/Assets/Resources/data/prediction"):
        print("Creating prediction directory...")
        os.makedirs("../UnityProject/Assets/Resources/data/prediction")
    for i, img in zip(range(start_index, end_index), imgs):
        image_dir = (
            "../UnityProject/Assets/Resources/data/prediction/" + str(i) + ".png"
        )
        PIL.Image.fromarray(np.uint8(np.squeeze(img * 255))).save(image_dir)


def predict_thread(m, m_id, start_index, end_index):
    """
    This thread takes a model and predicts image labels for the range of index given.
    ======
    Parameter:
    m - model
    m_id - model index number starting from 1
    start_index - starting index of image to predict
    end_index - ending index of image to predict
    """
    print("===Predicting Thread for model {} Start===".format(m_id))
    X, _ = play_data()
    print("Predicting Thread: Predicting images...")
    pred_Y = m.predict(X)
    print("Predicting Thread: Saving predictions...")
    save_prediction(pred_Y, start_index, end_index)
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
    pred_t = threading.Thread(target=predict_thread,args=(m, "1", 1, 20),daemon=True)
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


def main():
    # Initialize Model
    if state['pretrain_finished']: # Load existing model
        with open(DATA_DIR + "player_data.txt", "r") as f:
            state['current_img'] = int(f.read().strip())
            state['batch'] = (current_img-1)//10
            print("Player currently played {} images, the BATCH_NUM is set to {}".format(current_img, BATCH_NUM))
        m = 
    elif config['has_pretrain_data']: # Pretrain model
        print("Pre-training the model...")
        m = pre_train()
        write_state("pretrain_finished", "1")
    else: # Customized dataset may not have any label
        print("No pretrain data. Initializing UNET by default...")
        m = unet()

    train_model_by_batch(m, prune)


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True" # Tensorflow related issue
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Pruning prints too many message without this
    DATA_DIR = "../UnityProject/Assets/Resources/data/"
    config = load_config()
    state = load_state()
    main()
