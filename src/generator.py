import keras
import numpy as np
import random
import sys
import numpy as np
import threading
from PIL import Image
from constants import IMAGE_SIZE, S

def get_grid_index(x):
	r = int(np.floor(x))
	return r if r < 13 else 12

class DataGenerator(keras.utils.Sequence):
	def __init__(self, images, annotations, batch_size=32, shuffle=True, X_as_images=True):
		self.images = images
		self.annotations = annotations
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.X_as_images = X_as_images
		self.on_epoch_end()
		
	def __len__(self):
		return int(len(self.images)/self.batch_size)
		
	def _load_image(self, path):
		img = Image.open(path)
		img =  img.resize(IMAGE_SIZE, Image.ANTIALIAS)
		if img.mode != 'RGB':
			img = img.convert('RGB')
		return img
		
	def on_epoch_end(self):
		self.indexes = np.arange(len(self.images))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)
			
	def __getitem__(self, index):
		indexes = self.indexes[index*self.batch_size : (index + 1) * self.batch_size]
		if self.X_as_images:
			X = np.zeros((self.batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
		else:
			X = self.batch_size * ['q']
		Y = np.zeros((self.batch_size, S, S, 5))
		for i, r in enumerate(indexes):
			if self.X_as_images:
				X[i] = np.array(self._load_image(self.images[r]))/255.0
			else:
				X[i] = self.images[r]
			Y[i] = self._generate_truth(r)
		return X, Y
		
	def _generate_truth(self, index):
		y = np.zeros((S, S, 5))
		for i in range(S):
			for j in range(S):
				for a in self.annotations[index]:
					dx = IMAGE_SIZE[0] / S
					dy = IMAGE_SIZE[1] / S
					center_x = a[0] / dx
					center_y = a[1] / dy
					gridx = get_grid_index(center_x)
					gridy = get_grid_index(center_y)
					y[gridx][gridy][0] = 1.0
					y[gridx][gridy][1] = center_x
					y[gridx][gridy][2] = center_y
					y[gridx][gridy][3] = a[2] / dx
					y[gridx][gridy][4] = a[3] / dy
		return y