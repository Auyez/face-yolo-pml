from keras import backend as K
from keras import callbacks
from constants import S, ANCHOR_BOX, IMAGE_SIZE
import numpy as np
import math
from PIL import Image
import time
import sys
 
 
def sigmoid(x):
	return 1/(1 + math.exp(-x))

class Box():
	def __init__(self, center_x, center_y, width, height, conf):
		self.xmin = center_x - width / 2
		self.xmax = center_x + width / 2
		self.ymin = center_y - height / 2
		self.ymax = center_y + height / 2
		self.conf = conf
	def iou(self, box):
		xmin = max(self.xmin, box.xmin)
		xmax = min(self.xmax, box.xmax)
		ymin = max(self.ymin, box.ymin)
		ymax = min(self.ymax, box.ymax)
		w = xmax - xmin
		h = ymax - ymin
		inter = w * h
		union = self.area() + box.area() - inter
		return inter / union
	def area(self):
		return (self.xmax - self.xmin) * (self.ymax - self.ymin)
	def __str__(self):
		return 'x: {} y: {} w: {} h: {} c: {}'.format(self.xmin, self.ymin, (self.xmax - self.xmin), (self.ymax - self.ymin), self.conf)

#Evaluate standalone
def evaluate(model, generator, iou_threshold = 0.5, conf_threshold = 0.3, convert = True):
	d, a = extract_predictions(model, generator, conf_threshold, convert)
	return evaluate_detections(d, a, iou_threshold)
# code adapted from https://github.com/experiencor/keras-yolo2
# return value Array of detections, Array of annotations
# each instance of array contains Boxes for each image
# !!! Number of images = floor(#images/batch_size) * batch_size
def extract_predictions(model, generator, conf_threshold = 0.3, convert = True):
	size = generator.size()
	all_detections = [None for i in range(size)]
	all_annotations = [None for i in range(size)]

	s = time.time()
	number_of_batches = len(generator)
	for i in range(number_of_batches):
		x, y = generator[i]
		if not convert:
			images = [Image.open(path) for path in x]
			for im_index, img in enumerate(images):
				if img.mode != 'RGB':
					images[im_index] = img.convert('RGB')
			x = np.array([ np.array(im) for im in images])
			print(x.shape)
			p = model.predict(x)
		else:
			p = model.predict(x)
		#iterate over images in batch
		for j in range(generator.batch_size):
			#send one image from batch
			pred_boxes = to_boxes(p[j], conf_threshold, convert)
			true_boxes = to_boxes(y[j], conf_threshold, False)
			pred_boxes.sort(key=lambda x: x.conf)
			true_boxes.sort(key=lambda x: x.conf)
			# step is equal to Batch size and j is offset of image in batch
			all_detections[i * generator.batch_size + j] = pred_boxes
			all_annotations[i * generator.batch_size + j] = true_boxes
		p_bar = math.ceil(i/number_of_batches * 30)
		e = time.time()
		sys.stdout.write('\r{}/{} [{}{}] - {}s'.format(i, number_of_batches, p_bar * "=", (30 - p_bar) * ".", int(e - s)))
		sys.stdout.flush()
	return all_detections, all_annotations

# Computes AP for detections
# Input: array of detections for each image, array of annotations for each image
def evaluate_detections(all_detections, all_annotations, iou_threshold = 0.5):
	false_positives = np.zeros((0,))
	true_positives = np.zeros((0,))
	scores = np.zeros((0,))
	num_annotations = 0
	size = len(all_annotations)
	for i in range(size):
		num_annotations += len(all_annotations[i])
		detected = []
		for d in all_detections[i]:
			scores = np.append(scores, d.conf)
			best_iou = 0.0
			index = -1
			#Find highest IOU with ground truth values
			for j, annotation in enumerate(all_annotations[i]):
				predicted_iou = d.iou(annotation)
				if predicted_iou > best_iou:
					best_iou = predicted_iou
					index = j
			
			if best_iou >= iou_threshold and index not in detected:
				false_positives = np.append(false_positives, 0)
				true_positives = np.append(true_positives, 1)
				detected.append(index)
			else:
				false_positives = np.append(false_positives, 1)
				true_positives = np.append(true_positives, 0)
	indices = np.argsort(-scores)
	false_positives = np.cumsum(false_positives[indices])
	true_positives = np.cumsum(true_positives[indices])
	
	recall = true_positives / num_annotations
	precision = true_positives / (true_positives + false_positives)
	
	average_precission = compute_ap(recall, precision)
	return average_precission

# Get bounding boxes
# x,y,w,h defined relative to cell size (Absolute Width / Cell Width)
def to_boxes(prediction, conf, convert):
	boxes = []
	for i in range(S):
		for j in range(S):
			if convert:
				x = sigmoid(prediction[i][j][1]) + i
				y = sigmoid(prediction[i][j][2]) + j
				w = math.exp(prediction[i][j][3])# * ANCHOR_BOX[0]
				h = math.exp(prediction[i][j][4])# * ANCHOR_BOX[1]
				c = sigmoid(prediction[i][j][0])
			else:
				x = prediction[i][j][1]
				y = prediction[i][j][2]
				w = prediction[i][j][3]
				h = prediction[i][j][4]
				c = prediction[i][j][0]
			boxes.append(Box(x, y, w, h, c))
	return [box for box in boxes if box.conf >= conf]

#Code originally from https://github.com/rbgirshick/py-faster-rcnn.
def compute_ap(recall, precision):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap 
	
class ValidationCallback(callbacks.Callback):
	def __init__(self, validation_generator, iou_threshold = 0.3, conf_threshold = 0.3, max_count = 3, start_epoch = 60):
		self.generator = validation_generator
		self.iou = iou_threshold
		self.conf = conf_threshold
		self.decreasing = False
		self.count = 0
		self.max_count = max_count
		self.start_epoch = max(start_epoch-1, 0)
		self.best = float('-inf')
		
		self.monitor_file = open('monitor.txt', 'w')
		self.monitor_file.write('test\n')
		self.monitor_file.flush()
		
	def on_epoch_end(self, epoch, logs={}):
		self.monitor_file.write('Epoch: {} Loss: {}\n'.format(epoch, logs.get('loss')))
		self.monitor_file.flush()
		if epoch >= self.start_epoch:
			val = evaluate(self.model, self.generator, self.iou, self.conf)
			#detections, annotations = extract_predictions(self.model, self.generator, self.iou, self.conf)
			if val >= self.best:
				self.model.save_weights('weights/darknet_yolo.h5')
				self.best = val
				self.decreasing = False
				self.count = 0
			else:
				self.decreasing = True
				self.count += 1
			print('Validation score: ' + str(val))
			self.monitor_file.write('Validation score: ' + str(val) + '\n')
			self.monitor_file.flush()
		if self.decreasing and self.count >= self.max_count:
			self.model.stop_training = True
			print('Stop training')