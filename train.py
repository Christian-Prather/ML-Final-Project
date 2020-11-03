import os
import glob
import trimesh
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input, Conv1D
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np

DATA_DIR = "./data/"
BATCH_SIZE = 1
num_classes = 3
num_points = 2048

def parse_dataset():

    train_points = []
    train_labels = []
    folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))
    print(folders)

    for i, folder in enumerate(folders):
        train_files = glob.glob(os.path.join(folder, "train/*"))
        print(train_files)

        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

    return (
        np.array(train_points),
        np.array(train_labels)
    )

train_points, train_labels = parse_dataset()

train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
# train_dataset = tf.reshape(train_dataset,[2048,3])
train_dataset = train_dataset.shuffle(len(train_points)).batch(BATCH_SIZE)
print(train_points.shape)
print(train_dataset)

model = Sequential()
# model.add(keras.Input(shape=(num_points, 3)))
model.add(Dense(8, input_shape=(num_points, 3), activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# model.add(Conv1D(8, 3, input_shape=(num_points, 3), activation='relu'))
# model.add(Conv1D(num_classes, 3, activation='softmax'))


model.compile(optimizer=Adam(),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# model.summary()

model.fit(train_dataset, 
            epochs=20)
