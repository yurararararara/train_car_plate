from keras.preprocessing.image import ImageDataGenerator


from keras.utils import np_utils
from keras.models import Sequential
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping

import os
import pandas as pd
import numpy as np
import cv2
import argparse
from sklearn.metrics import accuracy_score
import sys
# 데이터 훈련하기
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# np.load = np_load_old
X_train, X_test, y_train, y_test = np.load('./numpy_data/train_data_set.npy')
print(X_train.shape)
print(X_train.shape[0])

#모델 정의
model = Sequential()
model.add(Dense(512, input_shape=(64, 64,3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))

model.add(Activation('softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

model.summary()

# hdf5_file = "./han-model.hdf5"

# if os.path.exists(hdf5_file):
#     model.load_weights(hdf5_file)

# else:
#     model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=600, epochs=10)
#     model.save_weights(hdf5_file)

def parse_args(args):
    parser = argparse.ArgumentParser(description='config info')
    parser.add_argument("--input_scale",default=416, type=int) # 320 + 32*n
    parser.add_argument("--batch_size",default=2, type=int)
    parser.add_argument("--box_per_grid",default=3, type=int)
    parser.add_argument("--class_num",default=7, type=int)  
    parser.add_argument("--epochs",default=1, type=int)   

    parser.add_argument("--train_dir_path",default="dataset/train_data", type=str)
    parser.add_argument("--test_dir_path",default="dataset/test_data", type=str)
    parser.add_argument("--class_name_file",default="dataset/class.names", type=str)

    parser.add_argument("--noobject_weight",default=.5, type=float)   
    parser.add_argument("--coord_weight",default=5. , type=float)

    args = parser.parse_args(args)
    args.grid_size = args.input_scale // 32


    return args

args = parse_args(sys.argv[1:])
train_loss = tf.keras.metrics.Mean()
test_loss = tf.keras.metrics.Mean()

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_function(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    loss = loss_function(labels, predictions)
    test_loss(loss)

txt = '에포크: {}, 스텝 : {}, 손실: {:.5f}'
for epoch in range(args.epochs):
    step = 1
    for image in X_train:
        for labels in y_train:
            train_step(image, labels)
            print(txt.format((epoch + 1), step ,train_loss.result()))
            step += 1

    for test_image in X_test:
        for test_labels in y_test:
            test_step(test_image, test_labels)  

    print(txt.format((epoch + 1),train_loss.result(),test_loss.result() ))
