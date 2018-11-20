from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Reshape, LeakyReLU, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, GlobalAveragePooling2D
import keras.applications as applications
from constants import S, IMAGE_SIZE
from loader import load_weights
import numpy as np
import sys


def create_model():
	return tiny_yolo_v2()

def mobile():
	mobile = applications.MobileNetV2(weights= "imagenet", include_top=False)
	
	input_layer = Input(shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
	output_layer = mobile(input_layer)
	output_layer = Conv2D(128, (3,3), strides=(1,1), padding='same', use_bias=False)(output_layer)
	output_layer = BatchNormalization()(output_layer)
	output_layer = LeakyReLU(alpha=0.1)(output_layer)
	
	
	output_layer = Conv2D(5, (1,1), strides=(1,1), padding='same', use_bias=False)(output_layer)
	output_layer = Reshape((S, S, 5))(output_layer)
	#output_layer = Flatten()(output_layer)
	#output_layer = Dense(2048)(output_layer)
	#output_layer = Dropout(0.5)(output_layer)
	#output_layer = LeakyReLU(alpha=0.1)(output_layer)
	#output_layer = Dense(S * S * 5)(output_layer)
	#output_layer = Reshape((S, S, 5))(output_layer)
	
	new_mobile = Model(input_layer, output_layer)
	new_mobile.summary()
	
	return new_mobile
	#sys.exit()
def vgg16():
	Vgg = applications.VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
	Vgg.summary()
	for layer in Vgg.layers[:-4]:
		layer.trainable = False
		
	model = Sequential()
	model.add(Vgg)
	model.add(Conv2D(512, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	
	model.add(Conv2D(128, (1,1), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	#30
	model.add(Conv2D(5, (1,1), strides=(1,1), padding='same'))
	model.add(Activation('linear'))
	model.add(Reshape((S, S, 5)))
	model.summary()
	return model
	
	model.add(Flatten())
	model.add(Dense(128))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	model.add(Dense(S * S * 5))
	model.add(Reshape((S, S, 5)))
	model.summary()
	return model

def tiny_yolo_v1():
	model = Sequential()
	model.add(Conv2D(16, (3,3), strides=(1,1), padding='same', use_bias=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1],3)))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(pool_size=(2,2)))
	
	
	model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(pool_size=(2,2)))
	
	
	model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(pool_size=(2,2)))
	
	
	model.add(Conv2D(128, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(pool_size=(2,2)))
	
	
	model.add(Conv2D(256, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(pool_size=(2,2)))
	
	model.add(Conv2D(512, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(pool_size=(2,2)))
	
	model.add(Conv2D(1024, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	
	model.add(Conv2D(256, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	
	model.add(Flatten())
	model.add(Dense(S * S * 5))
	model.add(Reshape((S, S, 5)))
	model.summary()
	return model
	
	
def tiny_yolo_v2():
	#scale = 1/8
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
	model.summary()
	return model
	####
	#30
	'''model.add(Conv2D(5, (1,1), strides=(1,1), use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	
	model.add(Flatten())
	model.add(Dense(S*S*5))
	model.add(Reshape((S, S, 5)))'''
	#model.summary()
	#[0, 4, 8, 12, 16, 20, 24]
	return model

#Darknet 19
def darknet19():
	model = Sequential()
	#0
	model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', use_bias=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1],3)))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(pool_size=(2,2)))
	#4
	model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(pool_size=(2,2)))
	#8
	model.add(Conv2D(128, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	#11
	model.add(Conv2D(64, (1,1), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	#14
	model.add(Conv2D(128, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(pool_size=(2,2)))
	#18
	model.add(Conv2D(256, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	#21
	model.add(Conv2D(128, (1,1), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	#24
	model.add(Conv2D(256, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(pool_size=(2,2)))
	#28
	model.add(Conv2D(512, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	#31
	model.add(Conv2D(256, (1,1), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	#34
	model.add(Conv2D(512, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	#37
	model.add(Conv2D(256, (1,1), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	#40
	model.add(Conv2D(512, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(pool_size=(2,2)))
	#44
	model.add(Conv2D(1024, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	#47
	model.add(Conv2D(512, (1,1), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	#50
	model.add(Conv2D(1024, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	#53
	model.add(Conv2D(512, (1,1), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	#56
	model.add(Conv2D(1024, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	#59 uncomment this block to create Darknet 19 classifier for 1000 classes
	'''model.add(Conv2D(1000, (1,1), strides=(1,1), padding='same'))
	model.add(GlobalAveragePooling2D())
	model.add(Activation('softmax'))
	model.summary()
	return model'''
	#Last layers for YOLO
	model.add(Conv2D(1024, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	
	model.add(Conv2D(1024, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	
	model.add(Conv2D(1024, (3,3), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	
	model.add(Conv2D(128, (1,1), strides=(1,1), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	#filter number is center_x + center_y + width + height + confidence
	model.add(Conv2D(4 + 1, (1,1), strides=(1,1), padding='same'))
	model.add(Activation('linear'))
	model.add(Reshape((S, S, 5)))
	model.summary()
	'''model.add(Flatten())
	model.add(Dense(4096))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	model.add(Dense(S * S * 5))
	model.add(Activation('linear'))
	model.add(Reshape((S, S, 5)))
	model.summary()'''
	#[0, 4, 8, 11, 14, 18, 21, 24, 28, 31, 34, 37, 40, 44, 47, 50, 53, 56]
	return model