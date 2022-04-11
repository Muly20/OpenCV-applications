import numpy as np
import cv2
import os
from DetectTrack_utils import *

"""
This application use a fusion of detection and tracking of a ball 
("sports ball" as defined in the COCO database) in a video. 
Detection is done using YOLOv3 pre-trained model while tracking is 
supported for some algorithms available in OpenCV (TLD, KCF, CSRT, MOSSE).
Once the ball is detected, tracking is performed for several frames until 
re-detection is initiated.
"""


# Load YOLO model for Detection
net = loadModel()

# get soccer ball class index from COCO database
class_index = classIndex('sports ball')

# set parameters
objectnessThresh = 0.5
confidenceThresh = 0.5
nmsThresh = 0.4
inWidth = 416
inHeight = 416

# use thresholds=None for default values
thresholds = (objectnessThresh, confidenceThresh, nmsThresh)

# set tracker type
tracker_types = ['TLD', 'KCF', 'CSRT', 'MOSSE']
TRACKER = 'CSRT'

# Open Video Object
cap = cv2.VideoCapture('soccer-ball.mp4')
START_FRAME = 0
print(f"Video resolution (W,H): ({cap.get(cv2.CAP_PROP_FRAME_WIDTH)},"
      f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)})")
print(f"Start with frame #{START_FRAME}: "
     f"{cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)}")

# Open video writer object
fileName = f'soccer-tracked-{TRACKER}.avi'
fps = cap.get(cv2.CAP_PROP_FPS)
codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
capw = cv2.VideoWriter(fileName,
                       fourcc=codec,
                       fps=fps,
                       frameSize=(inWidth, inHeight))
if not capw.isOpened():
    print("could not open file to write")

# flags for detection/tracking
DETECT = False
TRACK = False
count = 0
prior_hist = None
# max number of frames to track before performing detection again
MAX_TRACKING = 20

k = 0

# hit 'ESC' to exit
while k != 27:
    # extract and preprocess video frame
    ret, frame = cap.read()
    if not ret:
        print('Cannot read video file. Exiting...')
        cap.release()
        capw.release()
        cv2.destroyAllWindows()
        exit()

    # cropping original image to fit the YOLO's aspect ratio
    # (without distortion)
    # frame = frame[:, 280:-280, :]
    frame = cv2.resize(frame, (inWidth, inHeight))

    # if object was identified in previous frame than perform tracking
    # unless its more than MAX_TRACKING frames that the object has been
    # tracked, then re-detect object
    if DETECT or TRACK:
        # perform tracking, unless tracked for 20 frames in a raw
        ok, bbox = tracker.update(frame)
        # tracker.update() returns floats as bbox coordinates
        bbox = tuple(map(round, bbox))
        count += 1
        if ok and count < MAX_TRACKING:
            TRACK = True
            DETECT = False
            args = (bbox, TRACKER)
            frame = drawBox(frame, 'TRACKING', args)
        else:
            # lost tracking (try detection)
            TRACK = False

    if not TRACK:
        # perform detection
        bbox, confidence = detectObject(net,
                                        class_index,
                                        frame,
                                        (inWidth, inHeight),
                                        thresholds)

        if bbox:
            # initialize tracker with detected bbox (upon a 'clean' frame)
            tracker = initializeTracker(tracker_type=TRACKER)
            if tracker:
                ok = tracker.init(frame, bbox)
            else:
                ok = False
            if ok:
                DETECT = True
                count = 0
                # average two immediately-post-detection histograms
                prior_hist = bbox_hist(frame, bbox)

            # draw detected bounding box
            args = (bbox, confidence)
            frame = drawBox(frame, 'DETECTION', args)
        else:
            frame = drawBox(frame, 'NONE')

    if capw.isOpened():
        capw.write(frame)
    cv2.imshow('frame', frame)
    k = cv2.waitKey(50) & 0xFF

print(f"Ended at frame #{int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1}")
cap.release()
capw.release()
cv2.destroyAllWindows()
