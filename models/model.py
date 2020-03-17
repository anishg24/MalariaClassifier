import os
import keras as K
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.dirname(os.getcwd()) + "/data/"
BATCH_SIZE = 100
EPOCHS = 15
input_shape = (128, 128, 3)

X = np.load(os.path.join(DATA_DIR, "image_arrays.npy"), allow_pickle=True)
y = np.load(os.path.join(DATA_DIR, "label_arrays.npy"), allow_pickle=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

print("Loaded and split data...")

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

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

model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1 , validation_data=(X_test, y_test))

# score = model.evaluate(X_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# model.save("malariaclassifier.h5")
