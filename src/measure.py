# USAGE
# python measure.py --image images/image1.jpg


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

pixelSizeX = None
pixelSizeY = None

colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))

#-------------------------------------------------------------------------------
#Find the box to do calibration

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
args = vars(ap.parse_args())

# load our input image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 30, 100, apertureSize = 3)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

print("Number of contours in the image = {}".format(len(cnts)))

# loop over the contours individually
for (i, c) in enumerate(cnts):
	# if the contour is not sufficiently large, ignore it
	if len(c) < 10:
		continue
	if cv2.contourArea(c) < 100:
		continue

	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	cv2.drawContours(image, c, -1, (0, 255, 0), 2)
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break

cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

# apply the four point transform to obtain a top-down
# view of the original image
calibrator = four_point_transform(image, screenCnt.reshape(4, 2))

height, width, channels = calibrator.shape

if pixelSizeX is None:
	pixelSizeX = 60.0 / width  #6 cm = 60 mm
if pixelSizeY is None:
	pixelSizeY = 60.0 / height  #6 cm = 60 mm

#-------------------------------------------------------------------------------
#Analysis of outer dimensions of scintillator

# load our input image, convert it to grayscale, and blur it slightly
gray2 = cv2.cvtColor(calibrator, cv2.COLOR_BGR2GRAY)
gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged2 = cv2.Canny(gray2, 75, 200,apertureSize = 3)
edged2 = cv2.dilate(edged2, None, iterations=1)
edged2 = cv2.erode(edged2, None, iterations=1)

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts2 = cv2.findContours(edged2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts2 = imutils.grab_contours(cnts2)
cnts2 = sorted(cnts2, key = cv2.contourArea, reverse = True)[:5]

print("Number of contours inside the warped box = {}".format(len(cnts2)))

# loop over the contours individually
for c in cnts2:
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 100:
		continue
	# approximate the contour
	peri2 = cv2.arcLength(c, True)
	approx2 = cv2.approxPolyDP(c, 0.02 * peri2, True)
	x, y, w, h = cv2.boundingRect(approx2)

	cv2.drawContours(calibrator, c, -1, (0, 255, 0), 2)

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
	(int(x+w), int(y+(h/2)+15)), cv2.FONT_HERSHEY_SIMPLEX,
	0.65, (255, 255, 255), 2)

#Crop the image for next Analysis
scintillator = calibrator[y+10:y+h-10, x+10:x+w-10]

#-------------------------------------------------------------------------------
#Find two elipses

# load our input image, convert it to grayscale, and blur it slightly
gray3 = cv2.cvtColor(scintillator, cv2.COLOR_BGR2GRAY)
gray3 = cv2.GaussianBlur(gray3, (5, 5), 0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged3 = cv2.Canny(gray3, 75, 200,apertureSize = 3)
edged3 = cv2.dilate(edged3, None, iterations=1)
edged3 = cv2.erode(edged3, None, iterations=1)

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts3 = cv2.findContours(edged3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts3 = imutils.grab_contours(cnts3)
cnts3 = sorted(cnts3, key = cv2.contourArea, reverse = True)[:5]


print("Number of contours inside scintillator = {}".format(len(cnts3)))
#draw the contour (black color) on original image
ellipse1 = scale_contour(cnts3[0], x+10,y+10,1)
cv2.drawContours(calibrator, ellipse1, -1, (0,0,0), 2)
M1 = cv2.moments(ellipse1)
cx1 = int(M1['m10']/M1['m00'])
cy1 = int(M1['m01']/M1['m00'])
print("cx1 #{}:".format(cx1))
print("cy1 #{}:".format(cy1))

ellipse2 = scale_contour(cnts3[1], x+10,y+10,1)
cv2.drawContours(calibrator, ellipse2, -1, (0,0,0), 2)
M2 = cv2.moments(ellipse2)
cx2 = int(M2['m10']/M2['m00'])
cy2 = int(M2['m01']/M2['m00'])
print("cx2 #{}:".format(cx2))
print("cy2 #{}:".format(cy2))

#Make sure the order of the ellipses
if cx2<cx1:
	temp=cx1
	cx1=cx2
	cx2=temp
	temp=cy1
	cy1=cy2
	cy2=temp

colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),(255, 0, 255))
cv2.line(calibrator, (int(x), int(cy1)), (int(cx1), int(cy1)),colors[0], 2)
cv2.line(calibrator, (int(cx2), int(cy2)), (int(x+w), int(cy2)),colors[0], 2)
cv2.line(calibrator, (int(cx1), int(y)), (int(cx1), int(cy1)),colors[1], 2)
cv2.line(calibrator, (int(cx2), int(y)), (int(cx2), int(cy2)),colors[1], 2)
cv2.line(calibrator, (int(cx1), int(cy1)), (int(cx1), int(y+h)),colors[2], 2)
cv2.line(calibrator, (int(cx2), int(cy2)), (int(cx2), int(y+h)),colors[2], 2)
cv2.line(calibrator, (int(cx1), int(cy1)), (int(cx2), int(cy2)),colors[3], 2)

dA=dist.euclidean((x, cy1), (cx1, cy1))*pixelSizeX
dB=dist.euclidean((cx2, cy2), (x+w, cy2))*pixelSizeX
dC=dist.euclidean((cx1, y), (cx1, cy1))*pixelSizeY
dD=dist.euclidean((cx2, y), (cx2, cy2))*pixelSizeY
dE=dist.euclidean((cx1, cy1), (cx1, y+h))*pixelSizeY
dF=dist.euclidean((cx2, cy2), (cx2, y+h))*pixelSizeY
dG=dist.euclidean((cx1, cy1), (cx2, cy2))*pixelSizeX

(x1, y1) = midpoint((x, cy1), (cx1, cy1))
(x2, y2) = midpoint((cx2, cy2), (x+w, cy2))
(x3, y3) = midpoint((cx1, y), (cx1, cy1))
(x4, y4) = midpoint((cx2, y), (cx2, cy2))
(x5, y5) = midpoint((cx1, cy1), (cx1, y+h))
(x6, y6) = midpoint((cx2, cy2), (cx2, y+h))
(x7, y7) = midpoint((cx1, cy1), (cx2, cy2))



cv2.putText(calibrator, "{:.1f}mm".format(dA),
	(int(x1-50),int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX,
	0.65, (255, 255, 255), 2)
cv2.putText(calibrator, "{:.1f}mm".format(dB),
	(int(x2-50),int(y2-10)), cv2.FONT_HERSHEY_SIMPLEX,
	0.65, (255, 255, 255), 2)
cv2.putText(calibrator, "{:.1f}mm".format(dC),
	(int(x3+10),int(y3)), cv2.FONT_HERSHEY_SIMPLEX,
	0.65, (255, 255, 255), 2)
cv2.putText(calibrator, "{:.1f}mm".format(dD),
	(int(x4+10),int(y4)), cv2.FONT_HERSHEY_SIMPLEX,
	0.65, (255, 255, 255), 2)
cv2.putText(calibrator, "{:.1f}mm".format(dE),
	(int(x5+10),int(y5)), cv2.FONT_HERSHEY_SIMPLEX,
	0.65, (255, 255, 255), 2)
cv2.putText(calibrator, "{:.1f}mm".format(dF),
	(int(x6+10),int(y6)), cv2.FONT_HERSHEY_SIMPLEX,
	0.65, (255, 255, 255), 2)
cv2.putText(calibrator, "{:.1f}mm".format(dG),
	(int(x7-50),int(y7-10)), cv2.FONT_HERSHEY_SIMPLEX,
	0.65, (255, 255, 255), 2)


#-------------------------------------------------------------------------------
# show the original and scanned images
cv2.imshow("Original", image)
cv2.imshow("calibrator", calibrator)
cv2.imshow("Edged", edged)
cv2.imshow("Edged2", edged2)
cv2.imshow("gray", gray)
cv2.imshow("gray2", gray2)
cv2.imshow("scintillator", scintillator)
cv2.waitKey(0)
#cv2.destroyAllWindows()
