import argparse
import os
from sys import exit

import numpy as np
from matplotlib.image import imread
from skimage.transform import resize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_path", help="Path to the input file to predict off of.")
    p.add_argument("-r", "--retrain", help="Retrains a new model and runs the input file on the new model.",
                   action="store_true")
    p.add_argument("-b", "--batch_size", help="Custom batch size for retraining the model. Default 128", type=int)
    p.add_argument("-e", "--epochs", help="Custom epoch size for retraining the model. Default 12", type=int)
    p.add_argument("-t", "--test_size", help="Change the amount of validation data allowed for the model. Default 0.2",
                   type=float)
    args = p.parse_args()

    MODEL_PATH = "models/malariaclassifier.h5"
    INPUT_FILE = args.input_path
    BATCH_SIZE = args.batch_size if args.batch_size else 500
    EPOCHS = args.epochs if args.epochs else 15
    TEST_SIZE = args.test_size if args.test_size else 0.2

    if not os.path.exists(INPUT_FILE):
        print(f"File {INPUT_FILE} does not exist!")
        exit(2)

    from models.model import make_model

    if args.retrain:
        model = make_model(BATCH_SIZE, EPOCHS, TEST_SIZE)
    else:
        try:
            from keras.models import load_model
            model = load_model(MODEL_PATH)
        except OSError:
            print(f"Model not found at {MODEL_PATH}! Training a new model...")
            model = make_model(BATCH_SIZE, EPOCHS, TEST_SIZE)

    image_array = resize(imread(INPUT_FILE), (128, 128, 3))
    predictions = model.predict(np.array([image_array]))[0]
    pred_index = np.where(predictions == np.amax(predictions))[0][0]

    # The index will be either 0 or 1, where 0 is infected. Thanks to python, we can simplify the if statement.
    if pred_index:
        print("NOT INFECTED")
    else:
        print("INFECTED")


if __name__ == "__main__":
    main()