# OpenCV-applications

These are some sample applications of some basic vision and image manipulation tasks using OpenCV.

1. blemish_removal.py   - using OpenCV's mouse callback function perform patch replacement using border similarity and texture smoothness 
                          (minimum sobel-based gradients).
2. chroma_keying.py     - replaces green background with a template background using HSV colorspace, handling edges between front and back using sobel gradient and 
                          dilation for edge location and smoothing using median blur to handle green-leftovers. also applying highGUI for adjusting 
                          parameters and for backround color chosen by user using mouse callback function.
3. DetectTrack_main.py  - using a fusion of detection based on YOLOv3 and OpenCV's implementation of the CSRT tracker to track a soccer ball 
                          (or any other 'sports-ball') from a video.
4. DocumentAlignment.py - using homography to align a photographed document to fixed coordinates ('adobe scan'-like).
