import os
import argparse
import tensorflow as tf
import numpy as np

from tqdm import tqdm
from PIL import Image
from tensorflow.keras import backend as K

from model import build_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

seed = 42
np.random.seed = seed


def __get_arguments():
    parser = argparse.ArgumentParser(description="U-Net")

    parser.add_argument("--train_img_dir", type=str, required=True)
    parser.add_argument("--train_gt_dir", type=str, required=True)
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument("--stack_size", default=16, type=int)
    parser.add_argument("--train_img_num", default=20, type=int)
    parser.add_argument("--GPU", default=0, type=str)

    return parser.parse_args()


def __get_img(img_dir, stack_size):
    images = []
    patches = []
    file_names = next(os.walk(img_dir))[2]
    for i in tqdm(range(len(file_names))):
        img = Image.open(img_dir + file_names[i]).convert("RGB")
        img = img.resize((img.width // 2, img.height // 2))
        img = np.array(img).astype(np.float32) / 255.9
        patches.append(img)
        if i % stack_size == stack_size - 1:
            patches = np.asarray(patches).transpose(1, 2, 0, 3)
            images.append(patches)
            patches = []
    return np.array(images)


def main():
    args = __get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

    X_train = __get_img(args.train_img_dir, args.stack_size)
    Y_train = __get_img(args.train_gt_dir, args.stack_size)

    train_size = X_train.shape[0]
    batch_mask = np.random.choice(train_size, args.train_img_num)
    X_batch = X_train[batch_mask]
    Y_batch = Y_train[batch_mask]

    model = build_model(X_train.shape[1:])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    # model.summary()

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=f"{args.result_dir}/logs"),
    ]

    model.fit(
        X_batch,
        Y_batch,
        batch_size=128,
        epochs=args.epoch,
        callbacks=callbacks,
    )

    model.save(
        os.path.join(args.result_dir, "model.h5"),
        include_optimizer=True,
    )


if __name__ == "__main__":
    main()
