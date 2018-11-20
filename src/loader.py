from PIL import Image
from constants import IMAGE_SIZE, S
from random import shuffle
import numpy as np
import sys

PATH = 0
BOX_COUNT = 1
BOX = 2

# Taken from https://github.com/andersy005/keras-yolo
class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]
    
    def reset(self):
        self.offset = 4

def load_weights(model, path):
	weight_reader = WeightReader(path)
	#layers = [0, 4, 8, 11, 14, 18, 21, 24, 28, 31, 34, 37, 40, 44, 47, 50, 53, 56]
	layers = [0, 4, 8, 12, 16, 20, 24]
	for i in layers:
		conv_layer = model.layers[i]
		print(conv_layer)
		# Load weights of BatchNormalization Layer
		if i != 59:
			norm_layer = model.layers[i+1]
			size = np.prod(norm_layer.get_weights()[0].shape)
			
			beta = weight_reader.read_bytes(size)
			gamma = weight_reader.read_bytes(size)
			mean = weight_reader.read_bytes(size)
			var = weight_reader.read_bytes(size)
			norm_layer.set_weights([gamma, beta, mean, var])
		# Load weights if there are biases
		if len(conv_layer.get_weights()) > 1:
			bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
			kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
			kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
			kernel = kernel.transpose([2,3,1,0])
			conv_layer.set_weights([kernel, bias])
		# Load weights if there are no biases
		else:
			kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
			kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
			kernel = kernel.transpose([2,3,1,0])
			conv_layer.set_weights([kernel])
	print('Loaded weights from ' + path)
			

def load_image(path):
	img = Image.open(path)
	img =  img.resize(IMAGE_SIZE, Image.ANTIALIAS)
	if img.mode != 'RGB':
		img = img.convert('RGB')
	return img
	
def preload(file_list, images_folder):
	print('Processing input...')
	X = []
	Y = []
	f = open(file_list, 'r')
	state = PATH
	faces = 0
	max_faces = 0
	img_width = 0
	img_height = 0
	boxes = []
	#full_set = []
	for i, line in enumerate(f):
		#sys.stdout.write("Images loaded: %d\r" % (len(X) + 1) )
		#sys.stdout.flush()
		if state == PATH:
			#full_set.append(line)
			img = Image.open(images_folder + '/' + line.strip())
			img_width = img.width
			img_height = img.height
			X.append(images_folder + '/' + line.strip())
			state = BOX_COUNT			
		elif state == BOX_COUNT:
			#full_set[-1] += line
			max_faces = int(line)
			faces = 0
			state = BOX
		elif state == BOX:
			line = line.split()
			w_coef = IMAGE_SIZE[0] / img_width
			h_coef = IMAGE_SIZE[1] / img_height
			w = int(line[2]) * w_coef
			h = int(line[3]) * h_coef
			x_center = int(line[0]) * w_coef + w/2
			y_center = int(line[1]) * h_coef + h/2
			boxes.append((x_center, y_center, w, h))
			
			faces += 1
			if faces == max_faces:
				if faces > S * S:
					del X[-1]
				else:
					Y.append(boxes)
				boxes = []
				state = PATH
	print('Images loaded: {}  {}'.format(len(X), len(Y)))
	return X, Y

def augment(file_list, images_folder):
	print('Augmentation...')
	annotations = []
	f = open(file_list, 'r')
	state = PATH
	faces = 0
	max_faces = 0
	img_width = 0
	img_height = 0
	boxes = []
	X = []
	truth = open('WIDER_train_aug.txt', 'w')
	OUTPUT_DIR = 'WIDER_AUG'
	for i, line in enumerate(f):
		if state == PATH:
			img = Image.open(images_folder + '/' + line.strip())
			img_width = img.width
			img_height = img.height
			X.append(images_folder + '/' + line.strip())
			annotations.append('{}.jpg\n'.format(len(X) + 1))
			state = BOX_COUNT			
		elif state == BOX_COUNT:
			#full_set[-1] += line
			annotations[-1] += line
			max_faces = int(line)
			faces = 0
			state = BOX
		elif state == BOX:
			annotations[-1] += line
			line = line.split()
			w = int(line[2])
			h = int(line[3])
			x = int(line[0])
			y = int(line[1])
			boxes.append((x, y, w, h))
			
			faces += 1
			if faces == max_faces:
				truth.write(annotations[-1])
				img = Image.open(X[-1])
				number = len(X) + 1
				img.save('{}/{}.jpg'.format(OUTPUT_DIR, number))
				
				#Flip
				flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
				truth.write('{}{}.jpg\n'.format(number, 'f'))
				truth.write(str(len(boxes)) + '\n')
				for b in boxes:
					x = img_width - b[0] - b[2]
					truth.write('{} {} {} {} \n'.format(x, b[1], b[2], b[3]))
				flipped.save('{}/{}{}.jpg'.format(OUTPUT_DIR, number, 'f'))
				
				#90
				rotated = img.transpose(Image.ROTATE_90)		
				truth.write('{}{}.jpg\n'.format(number, 'r9'))
				truth.write(str(len(boxes)) + '\n')
				for b in boxes:
					x = b[1]
					y = img_width - b[0] - b[2]
					truth.write('{} {} {} {} \n'.format(x, y, b[3], b[2]))
				rotated.save('{}/{}{}.jpg'.format(OUTPUT_DIR, number, 'r9'))
				
				
				#180
				rotated = img.transpose(Image.ROTATE_180)
				truth.write('{}{}.jpg\n'.format(number, 'r18'))
				truth.write(str(len(boxes)) + '\n')
				for b in boxes:
					x = img_width - b[0] - b[2]
					y = img_height - b[1] - b[3]
					truth.write('{} {} {} {} \n'.format(x, y, b[2], b[3]))
				rotated.save('{}/{}{}.jpg'.format(OUTPUT_DIR, number, 'r18'))
				img.close()
				flipped.close()
				rotated.close()
				boxes = []
				state = PATH
	truth.close()
	print('Images loaded: {}'.format(len(X)))