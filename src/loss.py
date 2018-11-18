#import plaidml.keras
#plaidml.keras.install_backend()
from keras import backend as K
from constants import BATCH_SIZE, S, IMAGE_SIZE, ANCHOR_BOX
import numpy as np
import math
#0.006
OBJECT_SCALE = 1#5
NOOBJECT_SCALE = 0.5#1
COORD_SCALE = 5#1

def yolo_loss(y_true, y_pred):
	grid = np.zeros((BATCH_SIZE, S, S, 2))
	for k in range(BATCH_SIZE):
		for i in range(S):
				for j in range(S):
					grid[k][i][j] = np.array([i, j])
	pred_xy = K.sigmoid(y_pred[..., 1:3]) + K.variable(grid) #, dtype='float64') #
	pred_wh = K.exp(y_pred[..., 3:5]) * K.variable(np.reshape(ANCHOR_BOX, [1,1,1,2])) #, dtype='float64')
	pred_conf = K.sigmoid(y_pred[..., 0])
	
	true_xy = y_true[..., 1:3]
	true_wh = y_true[..., 3:5]
	true_wh_half = true_wh / 2
	true_mins = true_xy - true_wh_half
	true_maxes = true_xy + true_wh_half
	
	pred_wh_half = pred_wh / 2
	pred_mins = pred_xy - pred_wh_half
	pred_maxes = pred_xy + pred_wh_half
	
	inter_mins = K.maximum(pred_mins, true_mins)
	inter_maxes = K.minimum(pred_maxes, true_maxes)
	inter_wh = K.maximum(inter_maxes - inter_mins, 0)
	
	inter_areas = inter_wh[..., 0] * inter_wh[..., 1]
	true_areas = true_wh[..., 0] * true_wh[..., 1]
	pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

	union_areas = true_areas + pred_areas - inter_areas + K.epsilon()
	iou_scores = inter_areas/union_areas
	coord_mask = K.expand_dims(y_true[..., 0], axis=-1)
	coord_mask = K.repeat_elements(coord_mask, 2, 3)
	coord_mask = coord_mask * COORD_SCALE
	
	conf_mask = y_true[..., 0] * OBJECT_SCALE
	conf_mask = conf_mask + (1 - y_true[..., 0]) * NOOBJECT_SCALE
	
	loss_xy = K.sum(coord_mask * K.square(true_xy - pred_xy))
	loss_wh = K.sum(coord_mask * K.square(K.sqrt(true_wh) - K.sqrt(pred_wh)))
	#loss_wh = K.sum(coord_mask * K.square(true_wh - pred_wh))
	loss_conf_obj = K.sum(conf_mask * K.square(pred_conf - iou_scores))
	#r = coord_mask * K.square(true_wh - pred_wh)
	#return K.sum(true_wh[0])
	return (loss_xy + loss_wh + loss_conf_obj)/BATCH_SIZE