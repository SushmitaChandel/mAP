import mAP

import base_paths

def main():

	iou_threshold = 0.5
	path_pred_boxes = base_paths.base_path+'predictions/pred_boxes.csv'
	path_true_boxes = base_paths.base_path+'true/true_boxes.csv'
	pred_boxes,true_boxes = mAP.get_boxes(path_pred_boxes,path_true_boxes)
	average_precisions = mAP.mean_average_precision(pred_boxes, true_boxes, iou_threshold = iou_threshold, box_format="corners", num_classes=1)
	print(f'average precisions are {average_precisions}')

if __name__ == '__main__':
	main()