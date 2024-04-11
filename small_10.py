# USAGE
# python small.py --image images/small1.png


# import the necessary packages
from __future__ import print_function
from imagesearch.transform import four_point_transform
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from matplotlib import pyplot as plt
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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
args = vars(ap.parse_args())

#-------------------------------------------------------------------------------
#Find the box to do calibration

# load our input image, convert it to grayscale, and blur it slightly
first = cv2.imread(args["image"])

#resize --- Change fraction if needed
image = cv2.resize(first, (0, 0), fx = 0.5, fy = 0.5)

gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (1, 1), cv2.BORDER_DEFAULT)
gray1 = cv2.bilateralFilter(gray1, 100, 30, 10)

ret1, thresh1 = cv2.threshold(gray1,0,255,cv2.THRESH_TOZERO)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged1 = cv2.Canny(thresh1, 0, 20, apertureSize = 3)
#kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(100,100))
edged1 = cv2.dilate(edged1, None, iterations=1)
edged1 = cv2.erode(edged1, None, iterations=1)
#edged1 = cv2.morphologyEx(edged1, cv2.MORPH_OPEN, kernel1)

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts1 = cv2.findContours(edged1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnts1 = imutils.grab_contours(cnts1)
cnts1 = sorted(cnts1, key = cv2.contourArea, reverse = True)[:10]

print("Number of contours in the outer image = {}".format(len(cnts1)))

# loop over the contours individually
for (i, c) in enumerate(cnts1):
	# if the contour is not sufficiently large, ignore it
	if len(c) < 10:
		continue

	if cv2.contourArea(c) < 100000:
		continue

	# approximate the contour
	peri1 = cv2.arcLength(c, True)
	approx1 = cv2.approxPolyDP(c, 0.02 * peri1, True)

	print("Outer Contour Area = {}".format(cv2.contourArea(c)))
	print ("Length = {}".format(len(approx1)))
	#cv2.drawContours(image, c, -1, (0, 255, 0), 2)
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	#if len(approx1) == 4:
	screenCnt1 = approx1
    #	break

cv2.drawContours(image, [screenCnt1], -1, (0, 255, 0), 1)

# apply the four point transform to obtain a top-down
# view of the original image
calibrator = four_point_transform(image, screenCnt1.reshape(4, 2))

height, width, channels = calibrator.shape

if pixelSizeX is None:
	pixelSizeX = 54.0 / width  #6 cm = 54.2 mm
if pixelSizeY is None:
	pixelSizeY = 14.0 / height  #6 cm = 23.8 mm

#-------------------------------------------------------------------------------
#Analysis of outer dimensions of scintillator

# load our input image, convert it to grayscale, and blur it slightly
gray2 = cv2.cvtColor(calibrator, cv2.COLOR_BGR2GRAY)
gray2 = cv2.GaussianBlur(gray2, (3, 3), cv2.BORDER_DEFAULT)
gray2 = cv2.bilateralFilter(gray2, 100, 30, 10)

ret2, thresh2 = cv2.threshold(gray2,0,255,cv2.THRESH_TOZERO)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged2 = cv2.Canny(thresh2, 0, 50,apertureSize = 3)
edged2 = cv2.dilate(edged2, None, iterations=1)
edged2 = cv2.erode(edged2, None, iterations=1)

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts2 = cv2.findContours(edged2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cnts2 = imutils.grab_contours(cnts2)
cnts2 = sorted(cnts2, key = cv2.contourArea, reverse = True)[:1000]

print("Number of contours inside the warped box = {}".format(len(cnts2)))

# loop over the contours individually
for c in cnts2:
	# if the contour is not sufficiently large, ignore it
	if (cv2.contourArea(c) < 100000):
		continue
	# approximate the contour
	peri2 = cv2.arcLength(c, True)
	approx2 = cv2.approxPolyDP(c, 0.002 * peri2, True)
	x, y, w, h = cv2.boundingRect(approx2)
	rect = cv2.minAreaRect(c)
	(xloc, yloc), (width, height), angle = rect

	boxrect = cv2.boxPoints(rect)
	boxrect = np.int0(boxrect)

	#cv2.drawContours(calibrator, c, -1, (0, 255, 0), 2)

	print("Scintillator Contour Area = {}".format(cv2.contourArea(c)))
	print ("Sides = {}".format(len(approx2)))
	print ("xloc = {}".format(xloc))
	print ("yloc = {}".format(yloc))
	print ("width = {}".format(width))
	print ("height = {}".format(height))
	print ("angle = {}".format(angle))
	print ("x = {}".format(x))
	print ("y = {}".format(y))
	print ("w = {}".format(w))
	print ("h = {}".format(h))
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	#if len(approx2) == 4:
	screenCnt2 = approx2
	#	break

#cv2.drawContours(calibrator,[boxrect],0,(0,255,255),1)
cv2.rectangle(calibrator, (x, y), (x + w, y + h), (0, 255, 255), 1)

#rect = perspective.order_points(screenCnt2.reshape(4, 2))
#(tl, tr, br, bl) = rect
#cv2.line(calibrator, (int(tl[0]), int(tl[1])), (int(tr[0]), int(tr[1])),(0, 255, 255), 2)
#cv2.line(calibrator, (int(tr[0]), int(tr[1])), (int(br[0]), int(br[1])),(0, 255, 255), 2)
#cv2.line(calibrator, (int(br[0]), int(br[1])), (int(bl[0]), int(bl[1])),(0, 255, 255), 2)
#cv2.line(calibrator, (int(bl[0]), int(bl[1])), (int(tl[0]), int(tl[1])),(0, 255, 255), 2)

print ("pixelSizeX = {}".format(pixelSizeX))
print ("pixelSizeY = {}".format(pixelSizeY))

heightinmm = h*pixelSizeY
widthinmm  = w*pixelSizeX

# draw the object sizes on the image
cv2.putText(calibrator, "{:.3f}mm".format(widthinmm),
	(int(x+(w/2) - 15), int(y-15)), cv2.FONT_HERSHEY_SIMPLEX,
	0.65, (255, 255, 255), 2)
cv2.putText(calibrator, "{:.3f}mm".format(heightinmm),
	(int(x-10), int(y+(h/2)+15)), cv2.FONT_HERSHEY_SIMPLEX,
	0.65, (255, 255, 255), 2)

#Crop the image for next Analysis
scintillator = calibrator[y+100:y+h-100, x+100:x+w-100]


#-------------------------------------------------------------------------------
#Find two elipses

# load our input image, convert it to grayscale, and blur it slightly
gray3 = cv2.cvtColor(scintillator, cv2.COLOR_BGR2GRAY)
gray3 = cv2.GaussianBlur(gray3, (5, 5), cv2.BORDER_DEFAULT)
gray3 = cv2.bilateralFilter(gray3, 100, 30, 10)

ret,thresh3 = cv2.threshold(gray3,0,255,cv2.THRESH_TOZERO)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged3 = cv2.Canny(thresh3, 0, 20, apertureSize = 3)
#kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
edged3 = cv2.dilate(edged3, None, iterations=1)
#edged = cv2.erode(edged, kernel, iterations=1)
#edged3 = cv2.morphologyEx(edged3, cv2.MORPH_OPEN, kernel3)

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts3 = cv2.findContours(edged3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnts3 = imutils.grab_contours(cnts3)
cnts3 = sorted(cnts3, key = cv2.contourArea, reverse = True)[:10]


print("Number of contours inside scintillator = {}".format(len(cnts3)))
#draw the contour (black color) on original image

j=0

# create hull array for convex hull points
hull = []

for i in range(0, 2):
	cv2.drawContours(calibrator, c, -1, (0, 255, 0), 2)
	print("Area Ellipse={}".format(cv2.contourArea(cnts3[i])))

ellipse1 = scale_contour(cnts3[0], x+100,y+100,1)
cv2.drawContours(calibrator, ellipse1, -1, (0,0,0), 2)

(cx,cy),radius = cv2.minEnclosingCircle(ellipse1)
center = (int(cx),int(cy))
radius = int(radius)
#cv2.circle(calibrator,center,radius,(255,255,255),2)

xe1, ye1, we1, he1 = cv2.boundingRect(ellipse1)
cv2.rectangle(calibrator, (xe1, ye1), (xe1 + we1, ye1 + he1), (0, 255, 255), 1)

elip1box = cv2.fitEllipse(ellipse1)
(xel1,yel1),(mael1,MAel1),angle=elip1box
axes1 = elip1box[1]
minor1, major1 = axes1
cv2.ellipse(calibrator, elip1box,(255,0,0))


#First Funnel
cxinpix1=x+(12.5/pixelSizeX)
cyinpix1=y+h-(5/pixelSizeY)
radinner1=1.5/(2*pixelSizeX)
radouter1=4/(2*pixelSizeX)
center = (int(cxinpix1),int(cyinpix1))
radius = int(radinner1)
cv2.circle(calibrator,center,radius,(0,255,0),1)
radius = int(radouter1)
cv2.circle(calibrator,center,radius,(0,255,0),1)

M1 = cv2.moments(ellipse1)
cx1 = int(M1['m10']/M1['m00'])
cy1 = int(M1['m01']/M1['m00'])
print("cx1 #{}:".format(cx1))
print("cy1 #{}:".format(cy1))

ellipse2 = scale_contour(cnts3[1], x+100,y+100,1)
cv2.drawContours(calibrator, ellipse2, -1, (0,0,0), 2)

(cx,cy),radius = cv2.minEnclosingCircle(ellipse2)
center = (int(cx),int(cy))
radius = int(radius)
#cv2.circle(calibrator,center,radius,(255,255,255),2)

xe2, ye2, we2, he2 = cv2.boundingRect(ellipse2)
cv2.rectangle(calibrator, (xe2, ye2), (xe2 + we2, ye2 + he2), (0, 255, 255), 1)

elip2box = cv2.fitEllipse(ellipse2)
(xel2,yel2),(mael2,MAel2),angle=elip2box
axes2 = elip2box[1]
minor2, major2 = axes2
cv2.ellipse(calibrator, elip2box,(255,0,0))

#Second Funnel
cxinpix2=x+(37.5/pixelSizeX)
cyinpix2=y+h-(5/pixelSizeY)
radinner2=1.5/(2*pixelSizeX)
radouter2=4/(2*pixelSizeX)
center = (int(cxinpix2),int(cyinpix2))
radius = int(radinner2)
cv2.circle(calibrator,center,radius,(0,255,0),1)
radius = int(radouter2)
cv2.circle(calibrator,center,radius,(0,255,0),1)

M2 = cv2.moments(ellipse2)
cx2 = int(M2['m10']/M2['m00'])
cy2 = int(M2['m01']/M2['m00'])
print("cx2 #{}:".format(cx2))
print("cy2 #{}:".format(cy2))


hull.append(cv2.convexHull(ellipse1, False))
cv2.drawContours(calibrator, hull, 0, (240, 0, 159), 1, 8)
hull.append(cv2.convexHull(ellipse2, False))
cv2.drawContours(calibrator, hull, 1, (240, 0, 159), 1, 8)



#Make sure the order of the ellipses
if xel2<xel1:
	temp=xel1
	xel1=xel2
	xel2=temp
	temp=yel1
	yel1=yel2
	yel2=temp

colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),(255, 0, 255))
cv2.line(calibrator, (int(x), int(yel1)), (int(xel1), int(yel1)),colors[0], 2)
cv2.line(calibrator, (int(xel2), int(yel2)), (int(x+w), int(yel2)),colors[0], 2)
cv2.line(calibrator, (int(xel1), int(y)), (int(xel1), int(yel1)),colors[1], 2)
cv2.line(calibrator, (int(xel2), int(y)), (int(xel2), int(yel2)),colors[1], 2)
cv2.line(calibrator, (int(xel1), int(yel1)), (int(xel1), int(y+h)),colors[2], 2)
cv2.line(calibrator, (int(xel2), int(yel2)), (int(xel2), int(y+h)),colors[2], 2)
cv2.line(calibrator, (int(xel1), int(yel1)), (int(xel2), int(yel2)),colors[3], 2)

dA=dist.euclidean((x, yel1), (xel1, yel1))*pixelSizeX
dB=dist.euclidean((xel2, yel2), (x+w, yel2))*pixelSizeX
dC=dist.euclidean((xel1, y), (xel1, yel1))*pixelSizeY
dD=dist.euclidean((xel2, y), (xel2, yel2))*pixelSizeY
dE=dist.euclidean((xel1, yel1), (xel1, y+h))*pixelSizeY
dF=dist.euclidean((xel2, yel2), (xel2, y+h))*pixelSizeY
dG=dist.euclidean((xel1, yel1), (xel2, yel2))*pixelSizeX

(x1, y1) = midpoint((x, yel1), (xel1, yel1))
(x2, y2) = midpoint((xel2, yel2), (x+w, yel2))
(x3, y3) = midpoint((xel1, y), (xel1, yel1))
(x4, y4) = midpoint((xel2, y), (xel2, yel2))
(x5, y5) = midpoint((xel1, yel1), (xel1, y+h))
(x6, y6) = midpoint((xel2, yel2), (xel2, y+h))
(x7, y7) = midpoint((xel1, yel1), (xel2, yel2))



cv2.putText(calibrator, "{:.1f}mm".format(dA),
	(int(x1-50),int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX,
	0.65, colors[0], 2)
cv2.putText(calibrator, "{:.1f}mm".format(dB),
	(int(x2-50),int(y2-10)), cv2.FONT_HERSHEY_SIMPLEX,
	0.65, colors[0], 2)
cv2.putText(calibrator, "{:.1f}mm".format(dC),
	(int(x3+10),int(y3)), cv2.FONT_HERSHEY_SIMPLEX,
	0.65, colors[0], 2)
cv2.putText(calibrator, "{:.1f}mm".format(dD),
	(int(x4-80),int(y4)), cv2.FONT_HERSHEY_SIMPLEX,
	0.65, colors[0], 2)
cv2.putText(calibrator, "{:.1f}mm".format(dE),
	(int(x5+10),int(y5)), cv2.FONT_HERSHEY_SIMPLEX,
	0.65, colors[0], 2)
cv2.putText(calibrator, "{:.1f}mm".format(dF),
	(int(x6-80),int(y6)), cv2.FONT_HERSHEY_SIMPLEX,
	0.65, colors[0], 2)
cv2.putText(calibrator, "{:.1f}mm".format(dG),
	(int(x7-50),int(y7-10)), cv2.FONT_HERSHEY_SIMPLEX,
	0.65, colors[0], 2)




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
cv2.imshow("thresh3", thresh3)
cv2.imshow("gray3", gray3)
cv2.imshow("Edged3", edged3)
#plt.hist(gray1.ravel(),256,[0,256]); plt.show()
cv2.waitKey(0)
#cv2.destroyAllWindows()
