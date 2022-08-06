#!/usr/bin/python

import numpy as np
import random as rnd
import cv2
from utils import *
from make_model import *

def bb_intersection_over_union(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

seed = 11
rnd.seed(seed)
np.random.seed(seed)

videofile = './test_video.mp4'
cap = cv2.VideoCapture(videofile)

model = make_model()
model.load_weights('now_weight.h5')

lower = [0, 0, 0]
upper = [100, 100, 100]

stepSize = 60

lower = np.array(lower)
upper = np.array(upper)
count = 1
while(count<12):
    
    # ret,frame = cap.read()
    # print(frame.shape)
    # # print(img.shape)
    # if(ret == False):
    #     print("Done")
    #     break
    frame = cv2.imread('D:\\projects\\minor proj\\final\\CNN_Car_Detector\\eval_img\\frame'+str(count)+'.jpg')
    count+=1
    # cv2.imwrite("D:\\projects\\minor proj\\final\\CNN_Car_Detector\\eval_img\\frame%d.jpg" % count, frame)
    # count+=1
    #Convert image to HSV from BGR
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Find the pixels that correspond to road
    img_out = cv2.inRange(img_hsv, lower, upper)

    # Clean from noisy pixels and keep only the largest connected segment
    img_out = post_process(img_out)

    image_masked = frame.copy()

    # Get masked image
    image_masked[img_out == 0] = (0, 0, 0)
    s=0.25

    #Resize images for computational efficiency
    frame = cv2.resize(frame,None, fx=s,fy=s)
    image_masked = cv2.resize(image_masked,None, fx=s,fy=s)

    #Run the sliding window detection process
    bbox_list, totalWindows, correct, score = detectionProcess(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB), model, winH=50, winW=50, depth=3, nb_images=1, scale=1, stepSize=stepSize, thres_score=0.05)
    print(count, correct, score,bbox_list)
    count+=1
    #Draw the detections
    drawBoxes(frame, bbox_list)

    # Draw detections and road masks
    cv2.imshow('video',sidebyside(frame,image_masked))
    k = cv2.waitKey(3)

    #QUIT
    if(k & 0xFF == ord('q')):
        cv2.destroyWindow("video")
        break

print("Iou",stepSize/93 )
cap.release()
cv2.destroyAllWindows()
