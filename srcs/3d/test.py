import os
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imsave, imread
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")


def get_arguments():
    parser = argparse.ArgumentParser(description="CoSPA")

    parser.add_argument("--test_img_dir", type=str, required=True)
    parser.add_argument("--stack_size", default=16, type=int)
    parser.add_argument("--GPU", default=0, type=int)
    parser.add_argument("--result_dir", type=str, required=True)

    return parser.parse_args()


args = get_arguments()


def __get_img(img_dir, stack_size):
    images = []
    patches = []
    file_names = next(os.walk(img_dir))[2]
    for i in tqdm(range(len(file_names))):
        img = Image.open(img_dir + file_names[i]).convert("RGB")
        img = img.resize((img.width // 2, img.height // 2))
        img = np.array(img).astype(np.float32) / 255.0
        patches.append(img)
        if i % stack_size == stack_size - 1:
            patches = np.asarray(patches).transpose(1, 2, 0, 3)
            images.append(patches)
            patches = []
    return np.array(images)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    result_dir = args.result_dir

    X_test = __get_img(args.test_img_dir, args.stack_size)

    model = load_model(f"{result_dir}/model.h5", compile=False)

    preds_test = model.predict(X_test, verbose=1)

    os.makedirs(f"{result_dir}/pred/", exist_ok=True)
    os.makedirs(f"{result_dir}/binary/", exist_ok=True)
    for i in tqdm(range(preds_test.shape[0])):
        for k in range(preds_test.shape[3]):
            imsave(
                f"{result_dir}/pred/{str(i * preds_test.shape[3] + k)}.png",
                np.squeeze(preds_test[i, :, :, k, :]),
            )

            image = rgb2gray(
                imread(f"{result_dir}/pred/{str(i * preds_test.shape[3] + k)}.png")
            )
            thresh = threshold_otsu(image)
            binary = image > thresh
            imsave(
                f"{result_dir}/binary/{str(i * preds_test.shape[3] + k)}.png", binary
            )


if __name__ == "__main__":
    main()
