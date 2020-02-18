import os
import pandas as pd
import numpy as np
from skimage.transform import resize
from matplotlib.image import imread

DATA_DIR = "../data/"
IMAGE_DIR = DATA_DIR + "cell_images/"
INFECTED = IMAGE_DIR + "Parasitized/"
NOT_INFECTED = IMAGE_DIR + "Uninfected/"

print("Setup all imports and constants")

temp = {
    "Infected": [],
    "Not Infected": [],
    "Image Array": []
}

image_size = (128, 128, 3)

# These are infected confirmed
for infected_file in os.listdir(INFECTED):
    try:
        temp["Image Array"].append(resize(imread(NOT_INFECTED + not_infected_file), image_size))
    except:
        continue
    temp["Infected"].append(1)
    temp["Not Infected"].append(0)

print("Loaded infected files")

# These are uninfected
for not_infected_file in os.listdir(NOT_INFECTED):
    try:
        temp["Image Array"].append(resize(imread(NOT_INFECTED + not_infected_file), image_size))
    except:
        continue
    temp["Infected"].append(0)
    temp["Not Infected"].append(1)

print("Loaded uninfected files")

df = pd.DataFrame().from_dict(temp)

labels = df[df.columns[0:2]].values
images = np.stack(df["Image Array"].values)

np.save(DATA_DIR+"images.npy", images)
np.save(DATA_DIR+"labels.npy", labels)
print("Done")