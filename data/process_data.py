import numpy as np
import pandas as pd
from cv2 import imread, resize
import os
import matplotlib.pyplot as plt
from glob import glob

DATA_DIR = os.path.dirname(os.getcwd()) + "/data/"
INFECTED = os.path.join(DATA_DIR, "cell_images", "Parasitized", "C*.png")
NOT_INFECTED = os.path.join(DATA_DIR, "cell_images", "Uninfected", "C*.png")

# print("Setup all imports and constants")
# print(DATA_DIR)
# print(INFECTED)
# print(NOT_INFECTED)

def process():
    array_maker = lambda x: resize(imread(x), (128, 128))
    cutoff = 2500  # Final array size is cutoff * 2

    infected_arrays = np.array(list(map(array_maker, glob(INFECTED)[:cutoff])))
    uninfected_arrays = np.array(list(map(array_maker, glob(NOT_INFECTED)[:cutoff])))
    arrays = np.concatenate((infected_arrays, uninfected_arrays))
    print("Created image arrays")

    total_size = len(infected_arrays) + len(uninfected_arrays)
    infected_labels = np.zeros(total_size)
    uninfected_labels = np.zeros(total_size)
    infected_labels[:len(infected_arrays)] = 1
    uninfected_labels[len(infected_arrays):] = 1
    labels = np.stack((infected_labels, uninfected_labels), axis=1)
    print("Created labels")

    make_file = lambda x: os.path.join(DATA_DIR, x)
    np.save(make_file("image_arrays.npy"), arrays)
    np.save(make_file("label_arrays.npy"), labels)
    del arrays, labels
