import cv2
import numpy as np

"""
This program aligns a document within an image using homography transformation.
assumptions:
    1. document fully within the image (all corners)
    2. document is generally white on a darker background
    3. aspect ratio of A4 paper and portrait orientation
"""
# load image
image = cv2.imread('IMG_6253.jpg')

# resize the image to a (manageable) fixed size, keep aspect ratio
Hin, Win, _ = image.shape
image_aspect_ratio = Hin/Win

H = 1000
W = int(H/image_aspect_ratio)
image = cv2.resize(image, (W, H))

# binarize the image for contour finding
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, imageBin = cv2.threshold(imageGray, 200, 255, cv2.THRESH_BINARY)

# find external contours
contours, hierarchy = cv2.findContours(imageBin,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE
                                       )

# find the largest contour by area covered
area = np.asarray([cv2.contourArea(contour) for contour in contours])
contour_idx = np.argmax(area)
contour = contours[contour_idx]

# draw the document contour on the original image
imageContour = image.copy()
cv2.drawContours(imageContour, [contour], -1, (0, 255, 0), thickness=3)

# find enclosing rectangle approximation
# compute contour length for "error" between this length and the
# polygon approximation
p = cv2.arcLength(contour, closed=True)
srcPts = np.squeeze(cv2.approxPolyDP(contour, .02*p, closed=True))
srcPts = [tuple(pol) for pol in srcPts]

# draw polygon corners
imageCorners = image.copy()
for pnt in srcPts:
    cv2.circle(imageCorners, pnt, radius=3, color=(0,255,0), thickness=-1)

# exit if more than 4 corners
if len(srcPts) != 4:
    print("Error: largest polygon with more than 4 corners. Exiting...")
    exit()

# define destination points with aspect ratio of A4 paper, use 500 pixel width
Wout = 500
Hout = int(Wout * np.sqrt(2))

# change source points to numpy array as expected by "findHomography"
srcPts = np.asarray(srcPts)

# determine document orientation by comparing distance between points
dist1 = np.sqrt(np.sum(np.power(srcPts[0] - srcPts[1], 2)))
dist2 = np.sqrt(np.sum(np.power(srcPts[0] - srcPts[3], 2)))

# match source and destination points
if dist1 > dist2:
    dstPts = np.float32([[0,0], [0,Hout], [Wout,Hout], [Wout, 0]])
else:
    dstPts = np.float32([[Wout, 0], [0, 0], [0, Hout], [Wout, Hout]])

# compute homography matrix and transform input image to output image
M, _ = cv2.findHomography(srcPts, dstPts, cv2.RANSAC)
im_out = cv2.warpPerspective(image, M, (Wout,Hout))

# show results
# cv2.imshow('contour', imageContour)
# cv2.imshow('corners', imageCorners)
cv2.imshow('original', image)
cv2.imshow('out', im_out)
cv2.waitKey(0)
cv2.destroyAllWindows()