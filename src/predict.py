import plaidml.keras
plaidml.keras.install_backend()
from loader import preload
from model import create_model
from haar import HaarModel
from generator import DataGenerator
from loader import load_image
from PIL import Image, ImageDraw, ImageColor
from keras.models import load_model
from keras import backend as K
from loss import yolo_loss
from constants import S, IMAGE_SIZE, ANCHOR_BOX, BATCH_SIZE
from evaluate import evaluate
import random
import numpy as np
import math
import sys
import cv2

COLORS = ['red', 'green', 'blue', 'black', 'gray', 'orange', 'yellow', 'brown']
def sigmoid(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+np.exp(-x))
 
def convert_prediction(prediction):
	grid = np.zeros((prediction.shape[0], S, S, 2))
	for k in range(prediction.shape[0]):
		for i in range(S):
				for j in range(S):
						grid[k][i][j] = np.array([i, j])
					
	xy = sigmoid(prediction[..., 1:3]) + grid
	wh = np.exp(prediction[..., 3:5])# * np.reshape(ANCHOR_BOX, [1,1,1,2])
	conf = np.expand_dims(sigmoid(prediction[..., 0]), axis=-1)
	return np.block([conf, xy, wh])

def show_image(output, path = None, img = None):
	if path != None:
		img = Image.open(path)
	else:
		img = Image.fromarray(img)
	if img.mode != 'RGB':
		img = img.convert('RGB')
	d = ImageDraw.Draw(img)
	colors = COLORS[:]
	for i in range(S):
		for j in range(S):
			if output[i][j][0] > 0.3:
				print(output[i][j][0])
				width = output[i][j][3] * (img.width / S)
				height = output[i][j][4] * (img.height / S)
				x = output[i][j][1] * (img.width / S)
				y = output[i][j][2] * (img.height / S)
				if len(colors) == 0:
					colors = COLORS[:]
				k = random.randint(0, len(colors)-1)
				#if width * height > (img.width * 0.2) * (img.height * 0.2):
				d.rectangle([x - width/2, y - height/2, x + width/2, y + height/2], outline=ImageColor.getrgb(colors[k]))
				del colors[k]
	img.show()

def detect_on_image(path, model_path = None):
	if model_path == None:
		model = HaarModel()
		example = Image.open(path)
		img = np.zeros((1, example.height, example.width, 3))
	else:
		model = load_model(model_path, custom_objects={'yolo_loss': yolo_loss})
		img = np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
		example = load_image(path)
	print(path)
	img[0] = np.array(example) / 255.0
	P = model.predict(img)
	if model_path != None:
		P = convert_prediction(P)
	show_image(P[0], path)

def eval_model(model_path = 'models/model_final.rofl', convert = True):
	if convert:
		model = load_model(model_path, custom_objects={'yolo_loss': yolo_loss})
	else:
		model = model_path
	#X, Y = preload("WIDER_val.txt", "WIDER_train/images")
	#X, Y = preload("FDDB/FDDB_original.txt", "FDDB")
	X, Y = preload("WIDER_val/wider_face_val_bbx_gt.txt", "WIDER_val/images")
	training_generator = DataGenerator(X, Y, 1, True, convert)
	print(evaluate(model, training_generator, 0.5, 0.3, convert))
	

if len(sys.argv) > 1:
	if sys.argv[1] == 'eval':
		if sys.argv[2] == 'cv2':
			m = HaarModel()
			eval_model(m, False)
		else:
			eval_model(sys.argv[2])
	elif sys.argv[1] == 'model':
		print('Loading ' + sys.argv[2])
		detect_on_image(sys.argv[3], sys.argv[2])
	else:
		if sys.argv[1] == 'cv2':
			detect_on_image(sys.argv[2])
		else:
			detect_on_image(sys.argv[1], 'models/model.net')
