from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Reshape, LeakyReLU, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, GlobalAveragePooling2D
from constants import S, IMAGE_SIZE
from loader import load_weights
import numpy as np

def create_model():
	return tiny_yolo_v2()

def tiny_yolo_v2():
	model = Sequential()
	#0
	model.add(Conv2D(16, (3,3), strides=(1,1), padding='same', use_bias=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1],3)))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(pool_size=(2,2)))
	#4
	model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(pool_size=(2,2)))
	#8
	model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(pool_size=(2,2)))
	#12
	model.add(Conv2D(128, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(pool_size=(2,2)))
	#16
	model.add(Conv2D(256, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(pool_size=(2,2)))
	#20
	model.add(Conv2D(512, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	#model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	#24
	model.add(Conv2D(1024, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	#27
	model.add(Conv2D(256, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	#######
	model.add(Flatten())
	model.add(Dense(S * S * 5))
	model.add(Reshape((S, S, 5)))
	#model.summary()
	return model
	#[0, 4, 8, 12, 16, 20, 24]

