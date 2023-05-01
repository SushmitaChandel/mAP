import numpy as np

def intersection_over_union(boxes_preds, boxes_labels, box_format='midpoint'):

	"""
	Calculate intersection over union

	Parameters:
		boxes_preds (numpy): Predictions of Bounding boxesc (N X 4)
		boxes_labels (numpy): Correct labels of Bounding boxes (N X 4)
		box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

		Returns:
		numpy: Calculate intersection over union for all samples
	"""

	if box_format == "midpoint":
		box1_x1 = boxes_preds[...,0:1] - boxes_preds[...,2:3] / 2
		box1_y1 = boxes_preds[...,1:2] - boxes_preds[...,3:4] / 2
		box1_x2 = boxes_preds[...,0:1] + boxes_preds[...,2:3] / 2
		box1_y2 = boxes_preds[...,1:2] + boxes_preds[...,3:4] / 2
		box2_x1 = boxes_labels[...,0:1] - boxes_labels[...,2:3] / 2
		box2_y1 = boxes_labels[...,1:2] - boxes_labels[...,3:4] / 2
		box2_x2 = boxes_labels[...,0:1] + boxes_labels[...,2:3] / 2
		box2_y2 = boxes_labels[...,1:2] + boxes_labels[...,3:4] / 2		

	elif box_format == "corners":
		box1_x1 = boxes_preds[...,0:1]
		box1_y1 = boxes_preds[...,1:2]
		box1_x2 = boxes_preds[...,2:3]
		box1_y2 = boxes_preds[...,3:4] # (N,1)
		box2_x1 = boxes_labels[...,0:1]
		box2_y1 = boxes_labels[...,1:2]
		box2_x2 = boxes_labels[...,2:3]
		box2_y2 = boxes_labels[...,3:4]

	# elif box_format == "corner_width_height"

	x1 = np.maximum(box1_x1,box2_x1)
	y1 = np.maximum(box1_y1,box2_y1)
	x2 = np.minimum(box1_x2,box2_x2)
	y2 = np.minimum(box1_y2,box2_y2)

	intersection = np.clip(x2-x1,0,None)*np.clip(y2-y1,0,None)

	box1_area = np.abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
	box2_area = np.abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))

	return intersection/(box1_area + box2_area - intersection + 1e-6)

