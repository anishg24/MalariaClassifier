import os
import keras as K
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from data.process_data import process

DATA_DIR = os.path.dirname(os.getcwd()) + "/data/"
BATCH_SIZE = 500
EPOCHS = 15
input_shape = (128, 128, 3)
IMAGE_ARRAY = os.path.join(DATA_DIR, "image_arrays.npy")
LABEL_ARRAY = os.path.join(DATA_DIR, "label_arrays.npy")


def make_model(bs=BATCH_SIZE, e=EPOCHS, ts=0.2):
    try:
        X = np.load(IMAGE_ARRAY, allow_pickle=True)
        y = np.load(LABEL_ARRAY, allow_pickle=True)
    except FileNotFoundError:
        print(f"Processed data not found! Creating files at {IMAGE_ARRAY} and {LABEL_ARRAY}.")
        process()
        X = np.load(IMAGE_ARRAY, allow_pickle=True)
        y = np.load(LABEL_ARRAY, allow_pickle=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts)

    print("Loaded and split data...")

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss=K.losses.categorical_crossentropy,
                  optimizer=K.optimizers.Adam(),
                  metrics=['accuracy'])

    print("Created and compiled model...")

    model.fit(X, y, batch_size=bs, epochs=e, verbose=1, validation_data=(X_test, y_test))

    model.save("malariaclassifier.h5")
    return model
