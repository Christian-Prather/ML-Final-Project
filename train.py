import os
import glob
import trimesh
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input, Conv1D
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
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

# Save the model:

model.save('./frozen_keras.h5')

tf.saved_model.save(model, './frozen_models')

# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
# Get frozen ConcreteFunction                                                                                                                                   
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()
layers = [op.name for op in frozen_func.graph.get_operations()]

# Save frozen graph from frozen ConcreteFunction to hard drive                                                                                                  
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="frozen_graph.pb",
                  as_text=False)

