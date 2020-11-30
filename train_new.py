import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Softmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, LambdaCallback
import numpy as np
import os
import pathlib

DATA_DIR = "./data/"
BATCH_SIZE = 1
num_classes = 3
IMG_HEIGHT = 256 
IMG_WIDTH = 256 
cur_dir = os.getcwd()
train_data_dir = pathlib.Path(cur_dir + '/new_data')

# Class names (comes from directory names)
CLASS_NAMES = np.array([item.name for item in train_data_dir.glob('*') if (item.name != "place.txt") and (item.name!= ".DS_Store")])
print("Class names:", CLASS_NAMES)

###################################################################################
# Load images :

# Load training images
train_image_generator = ImageDataGenerator(rescale=1./255)
                                        #   rotation_range=20,
                                        #   shear_range=20)
                                        #   validation_split=0.2)
train_data_gen = train_image_generator.flow_from_directory(directory=str(train_data_dir),
                                                        #   batch_size=BATCH_SIZE,
                                                          subset="training",
                                                        #   color_mode="rgb",
                                                        #   shuffle=True,
                                                        #   target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                          classes = list(CLASS_NAMES))
                                                        #   class_mode = "sparse")

# # Load validation images
# validation_image_generator = ImageDataGenerator(rescale=1./255,
#                                               validation_split=0.2)
# validation_data_gen = validation_image_generator.flow_from_directory(directory=str(train_data_dir),
#                                                                     batch_size=batch_size,
#                                                                     subset="validation",
#                                                                     seed=seed,
#                                                                     color_mode="rgb",
#                                                                     shuffle=True,
#                                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
#                                                                     classes = list(CLASS_NAMES),
#                                                                     class_mode = "sparse")

# # Load test images
# test_image_generator =ImageDataGenerator(rescale=1./255)
# test_data_gen = test_image_generator.flow_from_directory(directory=str(test_data_dir),
#                                                         batch_size=1,
#                                                         color_mode="rgb",
#                                                         seed=seed,
#                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
#                                                         classes = list(CLASS_NAMES), 
#                                                         class_mode = "sparse")     
# test_steps_per_epoch = np.math.ceil(test_data_gen.samples / test_data_gen.batch_size)                       
# test_images = []
# test_labels = []
# for i in range(test_image_count):
#   (image, label) = next(test_data_gen)
#   test_images.append(image)
#   test_labels.append(label)

###################################################################################
# Callbacks:

# Stop early if training is not making progress
earlystopping = EarlyStopping(monitor = 'accuracy', 
                              verbose = 1,
                              min_delta = 0.01, 
                              patience = 5, 
                              mode = 'max',
                              restore_best_weights=True) 

callbacks_list = [earlystopping]

###################################################################################
# Model:

input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = MobileNetV2(input_shape = input_shape,
                        include_top = False,
                        weights='imagenet')        

base_model.trainable = False

model = Sequential([
  base_model,
  keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Dropout(0.25),
  keras.layers.Flatten(),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(num_classes, activation='softmax')
])

# model = Sequential()
# # model.add(keras.Input(shape=(num_points, 3)))
# model.add(Dense(8, input_shape=input_shape, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(64, activation='relu'))
# # model.add(Dense(128, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))
# # model.add(Conv1D(8, 3, input_shape=(num_points, 3), activation='relu'))
# # model.add(Conv1D(num_classes, 3, activation='softmax'))

model.compile(optimizer=Adam(),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# model.summary()

model.fit(train_data_gen, 
            epochs=25,
            callbacks=callbacks_list)

###################################################################################
# Save the model:

model.save('./frozen_keras.h5')

# tf.saved_model.save(model, './frozen_models')

# # Convert Keras model to ConcreteFunction
# full_model = tf.function(lambda x: model(x))
# full_model = full_model.get_concrete_function(
#         tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
# # Get frozen ConcreteFunction                                                                                                                                   
# frozen_func = convert_variables_to_constants_v2(full_model)
# frozen_func.graph.as_graph_def()
# layers = [op.name for op in frozen_func.graph.get_operations()]

# # Save frozen graph from frozen ConcreteFunction to hard drive                                                                                                  
# tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
#                   logdir="./frozen_models",
#                   name="frozen_graph.pb",
#                   as_text=False)