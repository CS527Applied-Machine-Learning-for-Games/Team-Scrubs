import os, sys, time, PIL, threading
import numpy as np
from model_prune import unet, prune_model
from data_loader import pretrain_data, test_data, play_data_by_batch, play_data_by_batch_x
from util import load_state, load_config, write_config, write_state

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler


def evaluate_thread(m, m_id, num_of_data):
    """
    This thread takes a model, evaluates it, and saves the result to report directory.
    ======
    Parameter:
    m - model
    m_id - model index number starting from 1
    """
    print("===Evaluating Thread for model {} Start===".format(m_id))
    print("Evaluating Thread: Loading testing dataset...")
    test_X, test_Y = test_data(num_of_data)
    print("Evaluating Thread: Evaluating model {}...".format(m_id))
    results = m.evaluate(test_X, test_Y, batch_size=10)
    print("Evaluating Thread: test loss, test acc:", results)
    if not os.path.exists(DATA_DIR+"reports"):
        print("Evaluating Thread: Creating reports directory...")
        os.makedirs(DATA_DIR+"reports")
    with open(DATA_DIR+"reports/{}.txt".format(m_id), "w") as f:
        f.write(str(results[1]))
    print("Evaluating Thread: Resut saved to {}reports/{}.txt".format(DATA_DIR, m_id))
    print("===Evaluating Thread for model {} End===".format(m_id))


def save_prediction(imgs, start_index, end_index):
    if not os.path.exists(DATA_DIR+"predictions"):
        print("Creating prediction directory...")
        os.makedirs(DATA_DIR+"predictions")
    for i, img in zip(range(start_index, end_index + 1), imgs):
        image_dir = DATA_DIR + "predictions/{}.png".format(i)
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
    pred_X = play_data_by_batch_x(start_index, end_index)
    print("Predicting Thread: Predicting images...")
    pred_Y = m.predict(pred_X)
    print("Predicting Thread: Saving predictions...")
    save_prediction(pred_Y, start_index, end_index)
    print("===Predicting Thread for model {} End===".format(m_id))


def pre_train():
    m = unet()
    print("Retrieving pre-train data...")
    X, Y = pretrain_data(config["num_of_pretrain_data"])
    H = m.fit(
        X,
        Y,
        batch_size=config["pretrain_batch_size"],
        epochs=config["pretrain_epochs"],
        validation_split=config["pretrain_split_percentage"]/100
    )
    print("Saving model...")
    m.save("./models/1.h5")
    print("Model saved to data/models/1.h5")
    evl_t = threading.Thread(target=evaluate_thread, args=(m, "1", config["num_of_test_data"]), daemon=True)
    evl_t.start()
    pred_t = threading.Thread(target=predict_thread, args=(m, "1", 1, config["pretrain_predict_batch"]*config["play_batch_size"]), daemon=True)
    pred_t.start()

    # Wait for both evaluation and prediction to finish
    evl_t.join()
    pred_t.join()
    return  m


def train_model_by_batch(m, BATCH_NUM):
    """
    Model pruning might be added to the process where every 50 drawings, we shrink the number of channels in the model by x%.
    The models after each training will be save to models folder.
    The evaluation result of each model will be save to results folder.
    The predictions will be saved to prediction
    """
    X, Y = play_data_by_batch((BATCH_NUM-1) * config["play_batch_size"] + 1, BATCH_NUM * config["play_batch_size"])
    for y in Y:
        print(y.shape)
        break
    Y = np.array([y[:, :, :, 0] for y in Y])
    H = m.fit(
        X, 
        Y, 
        batch_size=config["play_batch_size"], 
        epochs=config["play_epochs"]
    )
    if config["prune"] and BATCH_NUM % config["number_of_batch_to_prune"]:
        BATCH_NUM += 1
        m = prune_model(m, config["prune_percentage"]/100)
        mp.save("./models/{}.h5".format(BATCH_NUM))
        print("Pruned model saved to models/{}.h5".format(BATCH_NUM))
    else:
        BATCH_NUM += 1
        m.save("./models/{}.h5".format(BATCH_NUM))
        print("Model saved to models/{}.h5".format(BATCH_NUM))
        
    evl_t = threading.Thread(target=evaluate_thread, args=(m, str(BATCH_NUM), config["num_of_test_data"]), daemon=True)
    evl_t.start()
    start_index = (BATCH_NUM - 2 + config["pretrain_predict_batch"]) * config["play_batch_size"] + 1
    end_index = start_index + config["play_batch_size"] - 1
    if end_index < config["num_of_play_data"]:
        pred_t = threading.Thread(target=predict_thread, args=(m, str(BATCH_NUM), start_index, end_index), daemon=True)
        pred_t.start()
        pred_t.join()
    # Wait for both evaluation and prediction to finish
    evl_t.join()
    return m, str(BATCH_NUM)


def main():
    # Initialize Model
    if state["pretrain_finished"]: # Load existing model
        with open(DATA_DIR + "player_data.txt", "r") as f:
            state["current_img"] = int(f.read().strip())
            write_state("current_img", state["current_img"])
            print("Player currently played {} images, the BATCH_NUM is set to {}".format(state["current_img"], state["current_model"]))
        # TODO: Change this to load the latest model
        m = load_model("./models/{}.h5".format(state["current_model"]))
    else:
        if config["has_pretrain_data"]: # Pretrain model
            print("Pre-training the model...")
            m = pre_train()
        else: # Customized dataset may not have any label
            print("No pretrain data. Initializing UNET by default...")
            m = unet()
            m.save("./models/1.h5")
        write_state("pretrain_finished", 1)
        write_state("current_model", 1)
        state["current_model"] = 1
    # Every 10 seconds, it checks the drawing folder where the player's labeling will be saved.
    # Whenever there are greater of equal to `config["play_batch_size"]` new images in the drawing folder,
    # the model trains on a batch of `config["play_batch_size"]` images.
    batch_size = state["current_model"] * config["play_batch_size"]
    while True:
        index_ls = [int(x[0]) for x in [f.split(".") for f in os.listdir(DATA_DIR + "drawings/")] if len(x) == 2 and x[1] == "png"]
        print("Currently detect ", len(index_ls), "drawing(s).")
        if index_ls and max(index_ls) >= batch_size:
            m, state["current_model"] = train_model_by_batch(m, state["current_model"])
            write_state("current_model", state["current_model"])
        else:
            print("Waiting for more drawing ...")
            time.sleep(10)


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True" # Tensorflow related issue
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Pruning prints too many message without this
    DATA_DIR = "../UnityProject/Assets/Resources/data/"
    config = load_config()
    state = load_state()
    main()
