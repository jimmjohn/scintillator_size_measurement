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
gray = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT)
gray = cv2.bilateralFilter(gray, 100, 14, 100)


ret,thresh1 = cv2.threshold(gray,0,255,cv2.THRESH_TOZERO)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(thresh1, 10, 10*3, apertureSize = 3)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,10))
edged = cv2.dilate(edged, kernel, iterations=2)
#edged = cv2.erode(edged, kernel, iterations=1)
edged = cv2.morphologyEx(edged, cv2.MORPH_OPEN, kernel)
#edged = imutils.auto_canny(thresh1)

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:1000]

print("Number of contours in the image = {}".format(len(cnts)))

# loop over the contours individually
for (i, c) in enumerate(cnts):
	if cv2.contourArea(c) < 100:
		continue

	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	box = perspective.order_points(box)

	print("Contour Area = {}".format(cv2.contourArea(c)))
	cv2.drawContours(image,[box.astype("int")], -1, (0, 255, 0), 2)
	#cv2.drawContours(image, approx, -1, (0, 255, 0), 2)



# Select ROI
r = cv2.selectROI(image)

# Crop image
imCrop = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

# Display cropped image
cv2.imshow("ImageCroped", imCrop)


#-------------------------------------------------------------------------------
# show the original and scanned images
cv2.imshow("Original", image)
#cv2.imshow("calibrator", calibrator)
cv2.imshow("Edged", edged)
#cv2.imshow("Edges_bilatrel", edges)
cv2.imshow("gray", gray)
cv2.imshow("thresh1", thresh1)
cv2.waitKey(0)
#cv2.destroyAllWindows()
