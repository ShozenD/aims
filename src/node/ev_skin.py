import keras
import argparse
import tensorflow as tf
from keras.regularizers import l2
from keras.preprocessing import image
from keras.layers import Dense, BatchNormalization, Dropout, GlobalAveragePooling2D, Flatten

parser = argparse.ArgumentParser(description="image path")
parser.add_argument("img_path",
                    help='path to image')
args = parser.parse_args()
path = args.img_path
#set gpu
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

#set model
densenet = tf.keras.applications.DenseNet169(
    include_top = False,
    weights = "imagenet",
    input_shape = (224, 224, 3)
    )

x = GlobalAveragePooling2D()(densenet.output)
x = Dense(units = 256, activation = 'relu')(x)
x = Dropout(0.4)(x)
x = BatchNormalization()(x)
x = Dense(units = 128, activation = 'relu')(x)
x = Dropout(0.1)(x)
x = Dense(units = 64, activation = 'relu')(x)
pred = Dense(units = 7, activation = 'softmax', kernel_regularizer = l2(l = 0.01))(x)
model = keras.models.Model(inputs = densenet.input, outputs = pred)
model.load_weights('./densenet_skin_finetuned_best.h5')

x = model.layers[-8].output
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
output = Dense(7, activation='relu')(x)
model = keras.models.Model(inputs = model.input, outputs = output)
model.load_weights('./cp-0022.h5')
#compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae','accuracy'])
#load image
img = image.load_img(path,target_size=(224,224,3))
img = image.img_to_array(img)
img = img/255
img = img.reshape(1,224,224,3)

result=model.predict(img)[0]
print(
    result[0],
    result[1],
    result[2],
    result[3],
    result[4],
    result[5],
    result[6]
    )
sys.stdout.flush()
