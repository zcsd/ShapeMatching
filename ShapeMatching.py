import numpy as np
import cv2

print "Shape Matching Using Fourier Descriptor"

# Main loop
imgray = cv2.imread("a2.bmp", 0)
retvalth, imgthreshold = cv2.threshold(imgray, 50, 255, cv2.THRESH_BINARY)
imgthresholdNot = cv2.bitwise_not(imgthreshold)
#cv2.RETR_EXTERNAL,TREE
imgcontours, contours, hierarchy = cv2.findContours(imgthresholdNot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#print contours.size
imgdrawContours = np.zeros((imgray.shape[0],imgray.shape[1], 3), np.uint8)
cv2.drawContours(imgdrawContours, contours, -1, (255, 255, 255), 1)

for contour in contours:
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(imgdrawContours, center, radius, (0,255,0), 1)
    
#cv2.imshow("Original Gray", imgray)
#cv2.imshow("Threshold", imgthreshold)
#cv2.imshow("Contours", imgcontours)
imgcontourShow = cv2.resize(imgdrawContours, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
cv2.imshow("drawcontours", imgcontourShow)
cv2.waitKey(0)
cv2.destroyAllWindows()

