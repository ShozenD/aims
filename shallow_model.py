import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

#augmentation function
datagen = image.ImageDataGenerator(
    zoom_range=0.2,
    fill_mode="reflect",
    horizontal_flip=True
    )
#import csv
train = pd.read_csv('multi_label.csv')
train_image = []
#reshape images
for i in tqdm(range(train.shape[0])):
    img = image.load_img('images/'+str(train['file_id'][i])+'.jpg',target_size=(400,400,3))
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)

#create train&val dataset
X = np.array(train_image)
y = np.array(train.drop(['file_id'],axis=1))
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
#augment
it = datagen.flow(X_train, y_train)

#layer
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(400,400,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='relu'))
model.summary()
#compile
model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

#fit model
history = model.fit(it, epochs=10, validation_data=(X_test, y_test), batch_size=64)

#graph
epochs = range(len(loss))
loss = history.history["loss"]
val_loss = history.history["val_loss"]
plt.plot(epochs,loss,label="Training loss")
plt.plot(epochs,val_loss,label="Validation loss")
plt.show()
