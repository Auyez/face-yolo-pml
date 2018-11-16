from keras import backend as K
from keras import callbacks
from constants import S, ANCHOR_BOX, IMAGE_SIZE
import numpy as np
import math
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

			
# code adapted from https://github.com/experiencor/keras-yolo2	
def evaluate(model, generator, iou_threshold, conf_threshold):
	size = len(generator)
	all_detections = [None for i in range(size)]
	all_annotations = [None for i in range(size)]
	
	s = time.time()
	for i in range(size):
		x, y = generator[i]
		p = model.predict(x)
		pred_boxes = to_boxes(p, conf_threshold, False)
		#print(pred_boxes[0])
		true_boxes = to_boxes(y, conf_threshold, True)
		#print(true_boxes[0])
		pred_boxes.sort(key=lambda x: x.conf)
		true_boxes.sort(key=lambda x: x.conf)
		all_detections[i] = pred_boxes
		all_annotations[i] = true_boxes
	e = time.time()
	#print(e - s)
	false_positives = np.zeros((0,))
	true_positives = np.zeros((0,))
	scores = np.zeros((0,))
	num_annotations = 0
	
	s = time.time()
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
	e = time.time()
	#print(e - s)
	return average_precission

def to_boxes(p, conf, isTrue):
	prediction = p[0]
	boxes = []
	for i in range(S):
		for j in range(S):
			if not isTrue:
				x = sigmoid(prediction[i][j][1]) + i
				y = sigmoid(prediction[i][j][2]) + j
				w = math.exp(prediction[i][j][3]) * ANCHOR_BOX[0]
				h = math.exp(prediction[i][j][4]) * ANCHOR_BOX[1]
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