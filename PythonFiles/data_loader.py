import os, PIL
import numpy as np

image_root_dir = "../UnityProject/Assets/Resources/data/"

train_imgs_sub_dir = os.path.join(image_root_dir, "pretrain_imgs")
train_labels_sub_dir = os.path.join(image_root_dir, "pretrain_labels")
test_imgs_sub_dir = os.path.join(image_root_dir, "test_imgs")
test_labels_sub_dir = os.path.join(image_root_dir, "test_labels")
play_imgs_sub_dir = os.path.join(image_root_dir, "images")
play_labels_sub_dir = os.path.join(image_root_dir, "labels")
play_drawings_sub_dir = os.path.join(image_root_dir, "drawings")


def _get_data(img_root, label_root, start_index, end_index):
    X = []
    Y = []
    for i in range(start_index, end_index + 1):
        img_dir = os.path.join(img_root, str(i) + ".png")
        label_dir = os.path.join(label_root, str(i) + ".png")
        img = np.asarray(
            PIL.Image.open(img_dir).resize((256, 256), PIL.Image.ANTIALIAS)
        )
        mask = np.asarray(
            PIL.Image.open(label_dir).resize((256, 256), PIL.Image.ANTIALIAS)
        )
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        X.append(img)
        Y.append(mask)

    X = np.array(X)
    Y = np.array(Y)

    X = np.expand_dims(X, axis=3)
    Y = np.expand_dims(Y, axis=3)
    return X, Y


def _get_data_x(img_root, start_index, end_index):
    X = []
    for i in range(start_index, end_index + 1):
        img_dir = os.path.join(img_root, str(i) + ".png")
        img = np.asarray(
            PIL.Image.open(img_dir).resize((256, 256), PIL.Image.ANTIALIAS)
        )

        img = img / 255
        X.append(img)

    X = np.array(X)
    X = np.expand_dims(X, axis=3)

    return X


def pretrain_data(num_of_data):
    return _get_data(train_imgs_sub_dir, train_labels_sub_dir, 1, num_of_data)


def test_data(num_of_data):
    return _get_data(test_imgs_sub_dir, test_labels_sub_dir, 1, num_of_data)


def play_data_by_batch(start_index, end_index):
    return _get_data(
        play_imgs_sub_dir,
        play_drawings_sub_dir,
        start_index,
        end_index,
    )


def play_data_by_batch_x(start_index, end_index):
    return _get_data_x(
        play_imgs_sub_dir,
        start_index,
        end_index,
    )