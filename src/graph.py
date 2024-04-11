# USAGE
# python automate.py --image images/img6.png


# import the necessary packages
from __future__ import print_function
from imagesearch.transform import four_point_transform
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

#Shift contours bach to calibrator correct position
def scale_contour(cnt,x,y,scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [x+cx, y+cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

#Contour selection by mouse click
def mousecallback(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
		for i in range(0,cnts3.size):
			r=cv2.pointPolygonTest(cnts3[i],Point(y,x),False)
			if r>0:
				print("Selected contour "+i)



pixelSizeX = None
pixelSizeY = None

colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
args = vars(ap.parse_args())

#-------------------------------------------------------------------------------
#Find the box to do calibration

# load our input image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (5, 5), cv2.BORDER_DEFAULT)
gray1 = cv2.bilateralFilter(gray1, 100, 14, 100)

ret1, thresh1 = cv2.threshold(gray1,0,255,cv2.THRESH_TOZERO)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged1 = cv2.Canny(thresh1, 10, 100, apertureSize = 3)
#kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(100,100))
edged1 = cv2.dilate(edged1, None, iterations=1)
#edged = cv2.erode(edged, kernel, iterations=1)
#edged1 = cv2.morphologyEx(edged1, cv2.MORPH_OPEN, kernel1)

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts1 = cv2.findContours(edged1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts1 = imutils.grab_contours(cnts1)
cnts1 = sorted(cnts1, key = cv2.contourArea, reverse = True)[:1000]

print("Number of contours in the outer image = {}".format(len(cnts1)))

# loop over the contours individually
for (i, c) in enumerate(cnts1):
	# if the contour is not sufficiently large, ignore it
	if len(c) < 10:
		continue

	if cv2.contourArea(c) < 1000:
		continue

	# approximate the contour
	peri1 = cv2.arcLength(c, True)
	approx1 = cv2.approxPolyDP(c, 0.02 * peri1, True)

	print("Outer Contour Area = {}".format(cv2.contourArea(c)))

	cv2.drawContours(image, c, -1, (0, 255, 0), 2)
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	#if len(approx1) == 4:
    	screenCnt1 = approx1
    #        break



# This returns an array of r and theta values
lines = cv2.HoughLines(edged1,1,np.pi/180, 200)

# The below for loop runs till r and theta values
# are in the range of the 2d array
for r,theta in lines[0]:
	# Stores the value of cos(theta) in a
    a = np.cos(theta)
    # Stores the value of sin(theta) in b
    b = np.sin(theta)
    # x0 stores the value rcos(theta)
    x0 = a*r
    # y0 stores the value rsin(theta)
    y0 = b*r
    # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
    x1 = int(x0 + 1000*(-b))
    # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
    y1 = int(y0 + 1000*(a))
    # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
    x2 = int(x0 - 1000*(-b))
    # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
    y2 = int(y0 - 1000*(a))
	# cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
    # (0,0,255) denotes the colour of the line to be
    #drawn. In this case, it is red.
    cv2.line(image,(x1,y1), (x2,y2), (0,0,255),2)




#-------------------------------------------------------------------------------
# show the original and scanned images
cv2.imshow("Original", image)
cv2.imshow("thresh1", thresh1)
cv2.imshow("gray1", gray1)
cv2.imshow("Edged1", edged1)

cv2.waitKey(0)
#cv2.destroyAllWindows()
