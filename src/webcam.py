import plaidml.keras
plaidml.keras.install_backend()
import cv2
import numpy as np
from model import create_model
from keras.models import load_model
from loss import yolo_loss
import math
import sys
from PIL import Image
from predict import convert_prediction, show_image

dx = 640 / 7
dy = 480 / 7

def sigmoid(x, derivative=False):
  return 1/(1 + math.exp(-x))

def getBoxes(output):
	boxes = []
	for i in range(7):
		for j in range(7):
			conf = sigmoid(output[0][i][j][0])
			if conf > 0.5:
				x = dx * i + dx * sigmoid(output[0][i][j][1])
				y = dy * j + dy * sigmoid(output[0][i][j][2])
				w = dx * math.exp(output[0][i][j][3])
				h = dy * math.exp(output[0][i][j][4])
				boxes.append((int(x - w/2), int(y - h/2), int(w), int(h)))
	return boxes

video_capture = cv2.VideoCapture(0)
model = load_model('models/model.net', custom_objects={'yolo_loss': yolo_loss})
t = np.zeros((1, 448, 448, 3))
model.predict(t)

while True:
	ret, frame = video_capture.read()
	inp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	inp = cv2.resize(inp, (448, 448))
	batch = np.expand_dims(inp / 255.0, axis=0)
	output = model.predict(batch)
	for (x, y, w, h) in getBoxes(output):
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
	cv2.imshow('Video', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

video_capture.release()
cv2.destroyAllWindows()
