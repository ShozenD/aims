# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import keras
from keras.layers import Dense, BatchNormalization, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

datagen = image.ImageDataGenerator(
    rescale = 1./255,
    zoom_range=0.2,
    fill_mode='reflect',
    rotation_range=20,
    horizontal_flip=True
)

image_dir = "skin-images"
data_df = pd.read_csv('skin_condition_label1.csv')

data_df.head()

# +
# Parameters
BATCH_SIZE = 32
TARGET_SIZE = (224, 224)
INPUT_SHAPE= (224, 224, 3)

train_df, test_df = train_test_split(data_df, random_state = 80, test_size=0.2)

train_generator = datagen.flow_from_dataframe(
    train_df,
    directory = image_dir,
    x_col = "file_id",
    y_col = ["1_texture","1_pores","1_spot","1_saggy","1_clear","1_melanin","1_wrinkles"],
    target_size = TARGET_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = "raw"
)

test_generator = datagen.flow_from_dataframe(
    test_df,
    directory = image_dir,
    x_col = "file_id",
    y_col = ["1_texture","1_pores","1_spot","1_saggy","1_clear","1_melanin","1_wrinkles"],
    target_size = TARGET_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = "raw"
)
# -

plt.imshow(test_generator[0][0][3])
print(test_generator[0][1][3])

# +
densenet = tf.keras.applications.DenseNet169(
    include_top = False, 
    weights = "imagenet",
    input_shape = INPUT_SHAPE
    )
densenet.Trainable = False
x = GlobalAveragePooling2D()(densenet.output)
x = Dense(units = 256, activation = 'relu')(x)
x = Dropout(0.4)(x)
x = BatchNormalization()(x)
x = Dense(units = 128, activation = 'relu')(x)
x = Dropout(0.1)(x)
x = Dense(units = 64, activation = 'relu')(x)
pred = Dense(units = 7, activation = 'softmax', kernel_regularizer = l2(l = 0.01))(x)
model = keras.models.Model(inputs = densenet.input, outputs = pred)

model.load_weights('./tmp/finalresult/densenet_skin_finetuned_best.h5')
# -

x = model.layers[-8].output
x = Dense(256, activation='relu')(x)
x = BatchNormalization(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
output = Dense(7, activation='relu')(x)
model = keras.models.Model(inputs = model.input, outputs = output)

model.summary()

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

checkpoint_filepath = './tmp/checkpoint(fitting)/cp-{epoch:04d}.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    period=1,
    mode='auto',
    save_best_only=False)

# Fit model
history = newmodel.fit(
    train_generator,
    epochs = 20,
    validation_data = test_generator,
    callbacks=[model_checkpoint_callback]
  )

acc      = history.history['accuracy']
val_acc  = history.history['val_accuracy']
loss     = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))
plt.plot(epochs,loss,"b",label="Training loss")
plt.plot(epochs,val_loss,"g",label="validation loss")

plt.plot(epochs,acc,"b",label="Training accuracy")
plt.plot(epochs,val_acc,"g",label="validation accuracy")


