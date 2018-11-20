import numpy as np
import cv2
from constants import S, IMAGE_SIZE
from generator import get_grid_index
import sys

class HaarModel():
	def __init__(self):
		self.classifier = cv2.CascadeClassifier('src/haarcascade_frontalface_default.xml')
	def predict(self, batch):
		#batch = batch * 255
		B = batch.shape[0]
		out = np.zeros((B, S, S, 5))
		for i in range(B):
			try:
				img_height, img_width, ch = batch[i].shape
			except:
				print(batch[i].shape)
				sys.exit()
			img = batch[i].astype(np.uint8)
			gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			faces = self.classifier.detectMultiScale(
				gray,
				scaleFactor=1.1,
				minNeighbors=5,
				minSize=(30, 30)
			)
			for (x,y,w,h) in faces:		
				dx = img_width / S
				dy = img_height / S
				center_x = (x + w/2) / dx
				center_y = (y + h/2) / dy
				gridx = get_grid_index(center_x)
				gridy = get_grid_index(center_y)
				out[i][gridx][gridy][0] = 1.0
				out[i][gridx][gridy][1] = center_x
				out[i][gridx][gridy][2] = center_y
				out[i][gridx][gridy][3] = w / dx
				out[i][gridx][gridy][4] = h / dy
		return out