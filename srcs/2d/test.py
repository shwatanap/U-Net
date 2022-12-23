import os
import argparse
import numpy as np
import warnings

from skimage.io import imsave, imread
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")


def __get_arguments():
    parser = argparse.ArgumentParser(description="CoSPA")

    parser.add_argument("--test_img_dir", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--GPU", default=0, type=int)

    return parser.parse_args()


def __get_img(img_dir):
    images = []
    file_names = next(os.walk(img_dir))[2]
    for file_name in tqdm(file_names, total=len(file_names)):
        img = Image.open(img_dir + file_name).convert("RGB")
        img = np.array(img)
        images.append(img)
    return np.array(images)


def main():
    args = __get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    result_dir = args.result_dir

    X_test = __get_img(args.test_img_dir)

    model = load_model(f"{result_dir}/model.h5", compile=False)

    preds_test = model.predict(X_test, verbose=1)

    os.makedirs(f"{result_dir}/pred/", exist_ok=True)
    os.makedirs(f"{result_dir}/binary/", exist_ok=True)
    for i in tqdm(range(len(preds_test))):
        imsave(f"{result_dir}/pred/{str(i)}.png", np.squeeze(preds_test[i]))

        image = rgb2gray(imread(f"{result_dir}/pred/{str(i)}.png"))
        thresh = threshold_otsu(image)
        binary = image > thresh
        imsave(f"{result_dir}/binary/{str(i)}.png", binary)


if __name__ == "__main__":
    main()
