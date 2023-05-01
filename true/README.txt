The folder contains a .csv file named true_boxes.csv containing the bouding boxes in the following format:
boxes (list) : [[idx, class_pred, prob_score, x1, y1, x2, y2],...]].
The idx is founc using the image's name in a lexicographical fashion. 

For example, image 1 has name "1.jpg", image 2 has name "2.jpg",..., image 10 has name "10.jpg", them (image_name,idx) pairs are as follows:
(1.jpg,0)
(10.jpg,1)
(2.jpg,2)
(3.jpg,3)
(4.jpg,4)
(5.jpg,5)
(6.jpg,6)
(7.jpg,7)
(8.jpg,8)
(9.jpg,9)

