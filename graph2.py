import cv2
import numpy as np


filter = False


file_path = 'images/graph1.png'
img = cv2.imread(file_path)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
gray = cv2.bilateralFilter(gray, 100, 14, 100)

ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_TOZERO)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edges = cv2.Canny(thresh, 10, 100, apertureSize = 3)
#kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(100,100))
edges = cv2.dilate(edges, None, iterations=1)
#edged = cv2.erode(edged, kernel, iterations=1)
#edged1 = cv2.morphologyEx(edged1, cv2.MORPH_OPEN, kernel1)

cv2.imwrite('canny.jpg',edges)

lines = cv2.HoughLines(edges,1,np.pi/180,150)

if not lines.any():
    print('No lines were found')
    exit()

filter = True

if filter:
    rho_threshold = 100
    theta_threshold = 3.0

    # how many lines are similar to a given one
    similar_lines = {i : [] for i in range(len(lines))}
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i == j:
                continue

            rho_i,theta_i = lines[i][0]
            rho_j,theta_j = lines[j][0]
            if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                similar_lines[i].append(j)

    # ordering the INDECES of the lines by how many are similar to them
    indices = [i for i in range(len(lines))]
    indices.sort(key=lambda x : len(similar_lines[x]))

    # line flags is the base for the filtering
    line_flags = len(lines)*[True]
    for i in range(len(lines) - 1):
        if not line_flags[indices[i]]: # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
            continue

        for j in range(i + 1, len(lines)): # we are only considering those elements that had less similar line
            if not line_flags[indices[j]]: # and only if we have not disregarded them already
                continue

            rho_i,theta_i = lines[indices[i]][0]
            rho_j,theta_j = lines[indices[j]][0]
            if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                line_flags[indices[j]] = False # if it is similar and have not been disregarded yet then drop it now

print('number of Hough lines:', len(lines))

filtered_lines = []

if filter:
    for i in range(len(lines)): # filtering
        if line_flags[i]:
            filtered_lines.append(lines[i])

    print('Number of filtered lines:', len(filtered_lines))
else:
    filtered_lines = lines

for line in filtered_lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite('hough.jpg',img)

#-------------------------------------------------------------------------------
# show the original and scanned images
cv2.imshow("Original", img)
cv2.imshow("gray", gray)
cv2.imshow("Edges", edges)

cv2.waitKey(0)
#cv2.destroyAllWindows()
