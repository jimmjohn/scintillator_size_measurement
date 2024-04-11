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
gray1 = cv2.GaussianBlur(gray1, (3, 3), cv2.BORDER_DEFAULT)
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

	#cv2.drawContours(image, c, -1, (0, 255, 0), 2)
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx1) == 4:
            screenCnt1 = approx1
            break

cv2.drawContours(image, [screenCnt1], -1, (0, 255, 0), 2)

# apply the four point transform to obtain a top-down
# view of the original image
calibrator = four_point_transform(image, screenCnt1.reshape(4, 2))

height, width, channels = calibrator.shape

if pixelSizeX is None:
	pixelSizeX = 60.0 / width  #6 cm = 60 mm
if pixelSizeY is None:
	pixelSizeY = 60.0 / height  #6 cm = 60 mm

#-------------------------------------------------------------------------------
#Analysis of outer dimensions of scintillator

# load our input image, convert it to grayscale, and blur it slightly
gray2 = cv2.cvtColor(calibrator, cv2.COLOR_BGR2GRAY)
gray2 = cv2.GaussianBlur(gray2, (3, 3), cv2.BORDER_DEFAULT)
gray2 = cv2.bilateralFilter(gray2, 100, 14, 100)

ret2, thresh2 = cv2.threshold(gray2,0,255,cv2.THRESH_TOZERO)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged2 = cv2.Canny(thresh2, 10, 30,apertureSize = 3)
edged2 = cv2.dilate(edged2, None, iterations=1)
#edged2 = cv2.erode(edged2, None, iterations=1)

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts2 = cv2.findContours(edged2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts2 = imutils.grab_contours(cnts2)
cnts2 = sorted(cnts2, key = cv2.contourArea, reverse = True)[:100]

print("Number of contours inside the warped box = {}".format(len(cnts2)))

# loop over the contours individually
for c in cnts2:
	# if the contour is not sufficiently large, ignore it
	if (cv2.contourArea(c) < 25000) or (cv2.contourArea(c) > 26000):
		continue
	# approximate the contour
	peri2 = cv2.arcLength(c, True)
	approx2 = cv2.approxPolyDP(c, 0.001 * peri2, True)
	x, y, w, h = cv2.boundingRect(approx2)

	cv2.drawContours(calibrator, c, -1, (0, 255, 0), 2)

	print("Scintillator Contour Area = {}".format(cv2.contourArea(c)))
	print ("Length = {}".format(len(approx2)))

	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx2) == 4:
		screenCnt2 = approx2
		break

cv2.rectangle(calibrator, (x, y), (x + w, y + h), (0, 255, 0), 2)

heightinmm = h*pixelSizeY
widthinmm  = w*pixelSizeX

# draw the object sizes on the image
cv2.putText(calibrator, "{:.1f}mm".format(widthinmm),
	(int(x+(w/2) - 15), int(y-15)), cv2.FONT_HERSHEY_SIMPLEX,
	0.65, (255, 255, 255), 2)
cv2.putText(calibrator, "{:.1f}mm".format(heightinmm),
	(int(x-40), int(y+(h/2)+15)), cv2.FONT_HERSHEY_SIMPLEX,
	0.65, (255, 255, 255), 2)

#Crop the image for next Analysis
scintillator = calibrator[y+10:y+h-10, x+10:x+w-10]



#-------------------------------------------------------------------------------
# show the original and scanned images
cv2.imshow("Original", image)
cv2.imshow("calibrator", calibrator)
cv2.imshow("thresh1", thresh1)
cv2.imshow("gray1", gray1)
cv2.imshow("Edged1", edged1)
cv2.imshow("thresh2", thresh2)
cv2.imshow("gray2", gray2)
cv2.imshow("Edged2", edged2)

cv2.waitKey(0)
#cv2.destroyAllWindows()
