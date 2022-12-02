import os
import argparse
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    return parser.parse_args()


def prepare_image(file):
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (50, 50))
    img_arr = img_to_array(image)
    img_arr = img_arr / 255
    np_image = np.expand_dims(img_arr, axis=0)
    return np_image


def evaluate(model, dataset_path):
    image_files = os.listdir(dataset_path)
    for image_file_name in image_files:
        image_path = os.path.join(dataset_path, image_file_name)
        image = prepare_image(image_path)
        prediction_value = model.predict(image)

        if prediction_value < 0.5:
            print(f"{prediction_value}\t{image_path}\t\t-->\t\t is a cat")
        else:
            print(f"{prediction_value}\t{image_path}\t\t-->\t\t is a dog")


if __name__ == '__main__':
    args = parse_args()
    model = keras.models.load_model(args.model_input_path)
    evaluate(
        model=model,
        dataset_path=args.dataset_path
    )
