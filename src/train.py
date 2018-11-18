import plaidml.keras
plaidml.keras.install_backend()
from model import create_model
from evaluate import ValidationCallback
from keras.models import load_model
from loss import yolo_loss
from generator import DataGenerator
from loader import preload, load_image, load_weights
from keras import optimizers, callbacks
from keras import backend as K
from keras.applications import VGG16
from constants import S, IMAGE_SIZE, BATCH_SIZE
import numpy as np
import pdb
import pickle
import sys

class ChangeLearningRate(callbacks.Callback):
	def __init__(self):
		self.tuned = False
	def on_epoch_end(self, epoch, logs={}):
		#new_lr = K.get_value(self.model.optimizer.lr)
		loss = float(logs.get('loss'))
		#if epoch == 10:
		#	K.set_value(self.model.optimizer.lr, 1e-4)
		if epoch == 600:
			K.set_value(self.model.optimizer.lr, 1e-5)

#X, Y = preload("WIDER_train_aug.txt", "WIDER_AUG")
X, Y = preload("FDDB_train.txt", "FDDB_augmented")
training_generator = DataGenerator(X, Y, BATCH_SIZE)
X, Y = preload("FDDB/FDDB-val.txt", "FDDB")
#X, Y = preload("wider_val.txt", "WIDER_train/images")
validation_generator = DataGenerator(X, Y, 16)

adam = optimizers.Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
#adam = optimizers.SGD(lr=1e-4, decay=0.0005, momentum=0.9)
model = create_model()
#model.load_weights('weights/darknet_yolo.h5')
#load_weights(model, 'weights/darknet.weights')
#Freeze all classifier layers
#print(model.layers[-9])
#print(model.layers[-12])
#for layer in model.layers[:-12]:
#	layer.trainable = False
model.compile(loss = yolo_loss, optimizer=adam)
#model = load_model('model_final.rofl', custom_objects={'yolo_loss': yolo_loss})
callback = callbacks.ModelCheckpoint('weights/weights.{epoch:02d}.hdf',
									verbose=0,
									save_best_only=False,
									save_weights_only=True,
									mode='auto',
									period=5)
lr = ChangeLearningRate()
t_nan = callbacks.TerminateOnNaN()
#start validation on epoch 60
validation = ValidationCallback(validation_generator, 0.5, 0.3, 1300, 3)

#lr.last_weigths = 35
model.fit_generator(generator=training_generator,
					epochs=800,
					workers=8,
					max_queue_size=8,
					callbacks = [lr, validation, callback, t_nan],
					use_multiprocessing=False,
					initial_epoch = 0)
model.save('models/darknet-tiny.net')
