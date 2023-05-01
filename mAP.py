import numpy as np 
import iou
from collections import Counter
import pandas as pd 

def get_boxes(path_pred_boxes,path_true_boxes):

	try:
		df = pd.read_csv(path_pred_boxes,header=None)
	except pd.errors.EmptyDataError:
		df = pd.DataFrame()

	if df.empty:
		pred_boxes = []
	else:
		pred_boxes = df.values.tolist()

	try:
		df = pd.read_csv(path_true_boxes,header=None)
	except pd.errors.EmptyDataError:
		df = pd.DataFrame()

	if df.empty:
		true_boxes = []
	else:
		true_boxes = df.values.tolist()

	return pred_boxes,true_boxes



def mean_average_precision(
	pred_boxes, true_boxes, iou_threshold = 0.5, box_format="corners",
	num_classes = 20
 ):

	# pred_boxes (list) : [[train_idx, class_pred, prob_score, x1, y1, x2, y2],...]]
	average_precisions = []
	average_precisions_nfp = []
	epsilon = 1e-6
 
	s = 0
	for c in range(num_classes):
		detections = []
		ground_truths = []

		for detection in pred_boxes:
			if detection[1] == c:
				detections.append(detection)

		for true_box in true_boxes:
			if true_box[1] == c:
				ground_truths.append(true_box)

		# img 0 has 3 bboxes
		# img 1 has 5 bboxes
		# amount_bboxes = {0:3, 1:5}
		amount_bboxes = Counter([gt[0] for gt in ground_truths])

		for key, val in amount_bboxes.items():
			amount_bboxes[key] = np.zeros(val)

		# amount_bboxes = {0:np.array([0,0,0]), 1:np.array([0,0,0,0,0])}
		detections.sort(key=lambda x: x[2], reverse=True)
		TP = np.zeros((len(detections)))
		FP = np.zeros((len(detections)))
		total_true_bboxes = len(ground_truths)

		for detection_idx, detection in enumerate(detections):
			ground_truth_img = [
				bbox for bbox in ground_truths if bbox[0] == detection[0]
			]

			num_gts = len(ground_truth_img)
			best_iou = 0

			for idx, gt in enumerate(ground_truth_img):
				iou_ = iou.intersection_over_union(np.array(detection[3:]),
					np.array(gt[3:]),box_format=box_format)
				if iou_ > best_iou:
					best_iou = iou_
					best_gt_idx = idx
      
			if best_iou > iou_threshold:
				if amount_bboxes[detection[0]][best_gt_idx] == 0:
					TP[detection_idx] = 1
					amount_bboxes[detection[0]][best_gt_idx] = 1
				else:
					FP[detection_idx] = 1

			else:
				FP[detection_idx] = 1

		
		TP_cumsum = np.cumsum(TP, axis=0)
		FP_cumsum = np.cumsum(FP, axis=0)
		recalls = TP_cumsum / (total_true_bboxes + epsilon)
		precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
		precisions = np.concatenate((np.array([1]),precisions))
		recalls = np.concatenate((np.array([0]),recalls))
		average_precisions.append(np.trapz(precisions, recalls))

	return average_precisions