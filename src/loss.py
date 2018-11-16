#import plaidml.keras
#plaidml.keras.install_backend()
from keras import backend as K
from constants import BATCH_SIZE, S, IMAGE_SIZE, ANCHOR_BOX
import numpy as np
import math
#0.006
OBJECT_SCALE = 5#5
NOOBJECT_SCALE = 1#1
COORD_SCALE = 1#1

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

	
'''
def yolo_loss(y_true, y_pred):
	pred_xy = y_pred[..., 1:3]
	pred_wh = y_pred[..., 3:5]
	
	true_xy = y_true[..., 1:3]
	true_wh = y_true[..., 3:5]
	
	offsets = np.zeros((16, 7, 7, 2))
	grid = np.zeros((7,7,2))
	for i in range(S):
		for j in range(S):
			grid[i][j] = np.array([i*64, j * 64])
	for k in range(16):
		offsets[k] = grid
	#return K.eval(K.variable(offsets[..., 1]))
	real_pred_xy = K.variable(offsets) + pred_xy * 64
	real_pred_wh = pred_wh * 448
	real_true_xy = K.variable(offsets) + true_xy * 64
	real_true_wh = true_wh * 448
	#return pred_xy
	
	
	real_true_wh_half = real_true_wh / 2
	true_mins = real_true_xy - real_true_wh_half
	true_maxes = real_true_xy + real_true_wh_half
	
	real_pred_wh_half = real_pred_wh / 2
	pred_mins = real_pred_xy - real_pred_wh_half
	pred_maxes = real_pred_xy + real_pred_wh_half
	
	inter_mins = K.maximum(pred_mins, true_mins)
	inter_maxes = K.minimum(pred_maxes, true_maxes)
	inter_wh = K.maximum(inter_maxes - inter_mins, 0)
	#return K.eval(inter_wh)
	inter_areas = inter_wh[..., 0] * inter_wh[..., 1]
	#return K.eval(y_true[..., 0] * inter_areas)
	true_areas = real_true_wh[..., 0] * real_true_wh[..., 1]
	pred_areas = K.maximum(real_pred_wh[..., 0] * real_pred_wh[..., 1], 0)
	inter_areas = K.maximum(inter_areas, 0)
	#return K.eval(inter_areas)

	conf_mask = y_true[..., 0]
	union_areas = true_areas + pred_areas - inter_areas + K.epsilon()
	iou_scores = inter_areas/union_areas
	#return K.eval(iou_scores)
	coord_mask = K.expand_dims(y_true[..., 0], axis=-1)
	coord_mask = K.repeat_elements(coord_mask, 2, 3)
	loss_xy = K.sum(coord_mask * K.square(true_xy - pred_xy)) * lambdaC
	loss_wh = K.sum(coord_mask * K.square(K.sqrt(true_wh) - K.sqrt(pred_wh))) * lambdaC
	#return K.eval(loss_wh)
	pred_confidence = y_pred[..., 0]
	
	conf_mask = conf_mask + (1 - y_true[..., 0]) * lambdaN
	loss_conf_obj = K.sum(K.square(pred_confidence - iou_scores) * conf_mask)
	return (loss_xy + loss_wh + loss_conf_obj)/16.0
	#return [ K.eval(l) for l in [loss_xy, loss_wh, loss_conf_obj] ]'''