# mAP
The repository can be used to calculate average precision.

Source: https://www.youtube.com/watch?v=FppOzcDvaDI

Steps to use the package:
1. Add true.csv file in the folder "true". It contains the list of all true bounding boxes in the following format:
   [[train_idx, class_pred, prob_score, x1, y1, x2, y2],...]]
 You can put prob_score equals to 1 in case of true boxes.
2. Add pred.csv file in the folder "predictions". It contains the list of all true bounding boxes in the following format:
   [[train_idx, class_pred, prob_score, x1, y1, x2, y2],...]]
3. Update base_path in base_paths.py file and point to the current directory.
4. RUN demo.py file.
  


