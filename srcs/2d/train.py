import os
import argparse
import tensorflow as tf
import numpy as np

from tqdm import tqdm
from PIL import Image

from model import build_model

seed = 42
np.random.seed = seed


def __get_arguments():
    parser = argparse.ArgumentParser(description="U-Net")

    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument("--train_img_num", default=20, type=int)
    parser.add_argument("--GPU", default=0, type=int)

    return parser.parse_args()


def __get_img(img_dir):
    images = []
    file_names = next(os.walk(img_dir))[2]
    for file_name in tqdm(file_names, total=len(file_names)):
        img = Image.open(img_dir + file_name).convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        images.append(img)
    return np.array(images)


def main():
    args = __get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    epoch = args.epoch
    result_dir = args.result_dir

    X_train = __get_img(args.img_dir)
    Y_train = __get_img(args.gt_dir)

    train_size = X_train.shape[0]
    batch_mask = np.random.choice(train_size, args.train_img_num)
    X_batch = X_train[batch_mask]
    Y_batch = Y_train[batch_mask]

    model = build_model()
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    # model.summary()

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=f"{result_dir}/logs"),
    ]

    model.fit(
        X_batch,
        Y_batch,
        validation_split=0.3,
        batch_size=128,
        epochs=epoch,
        callbacks=callbacks,
    )

    model.save(
        os.path.join(result_dir, "model.h5"),
        include_optimizer=True,
    )


if __name__ == "__main__":
    main()
