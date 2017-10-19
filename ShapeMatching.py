import numpy as np
import cv2

print "Shape Matching Using Fourier Descriptor"

rect = (0, 0, 1, 1)
ix, iy = -1, -1
temSeleteFlag = False
temReadyFlag = False
matchOverFlag = False
temConfirmFlag = False
templeteVector = []
sampleVectors = []
sampleContours = []

def selectTemplete(event, x, y, flags, param):
    global rect, temSeleteFlag, temReadyFlag, ix, iy

    if event == cv2.EVENT_LBUTTONDOWN and temReadyFlag == False:
        temSeleteFlag = True
        ix, iy = x, y
       
    elif event == cv2.EVENT_LBUTTONUP:
        if temReadyFlag == False and temSeleteFlag == True:
            rect = (min(ix,x), min(iy,y), abs(ix-x), abs(iy-y))
            cv2.rectangle(imgOri, (ix,iy), (x,y), (255,0,0), 2)
        temSeleteFlag = False
        temReadyFlag = True

def getContours(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retvalth, imgthreshold = cv2.threshold(imgray, 50, 255, cv2.THRESH_BINARY)
    imgthresholdNot = cv2.bitwise_not(imgthreshold)
    kernel = np.ones((5,5), np.uint8)
    imgdilation = cv2.dilate(imgthresholdNot, kernel, iterations=2)
    imgcontours, contours, hierarchy = cv2.findContours(imgdilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #imgdrawContours = np.zeros((imgray.shape[0],imgray.shape[1], 3), np.uint8)
    #cv2.drawContours(imgdrawContours, contours, -1, (255, 255, 255), 1)
                
    #cv2.imshow("Original Gray", imgray)
    #cv2.imshow("Threshold", imgthreshold)
    #cv2.imshow("Contours", imgcontours)
    #imgcontourShow = cv2.resize(imgdrawContours, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    #cv2.imshow("drawcontours", imgcontourShow)
    return contours

def getTempleteCV():
    templeteROI = imgOricpy[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
    tpContour = getContours(templeteROI)

    for cnt in tpContour:
        x, y, w, h = cv2.boundingRect(cnt)
        for point in cnt:
            templeteVector.append( complex(point[0][0]-x, point[0][1]-y) )

def getSampleCV(spContour):
    x, y, w, h = cv2.boundingRect(spContour)
    sampleVector = []
    for point in spContour:
        sampleVector.append( complex(point[0][0]-x, point[0][1]-y) )
    sampleVectors.append(sampleVector)
    sampleContours.append(spContour)

def getsampleFTs():
    FTs = []
    for sampleVector in sampleVectors:
        sampleFT = np.fft.fft(sampleVector)
        FTs.append(sampleFT)

    return FTs    

def match(tp, sps):
    pass

# Main loop
imgOri = cv2.imread("a2.bmp", 1)
imgOricpy = imgOri.copy()
cv2.namedWindow("Original Image")
cv2.setMouseCallback("Original Image", selectTemplete)

while(True):
    cv2.imshow("Original Image", imgOri)
    
    if temReadyFlag == True and matchOverFlag == False and temConfirmFlag == True:
        getTempleteCV();
        contours = getContours(imgOricpy)
        cv2.drawContours(imgOri, contours, -1, (0, 0, 255), 1)
        for contour in contours:
            getSampleCV(contour);
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(imgOri, (x,y), (x+w,y+h), (0,255,0), 1)
            #rect = cv2.minAreaRect(contour)
            #box = cv2.boxPoints(rect)
            #box = np.int0(box)
            #cv2.drawContours(imgOri, [box],0,(0,255,0),1)
        #print sampleVectors[1]

        tpFT = np.fft.fft(templeteVector)
        sampleFTs = getsampleFTs()
        match(tpFT, sampleFTs)
        
        matchOverFlag = True
        
    key = cv2.waitKey(1) & 0xFF
    if key == ord('y') or key == ord('Y'):
        temConfirmFlag = True
    elif key == ord('q') or key == ord('Q'):    
        break
 
cv2.destroyAllWindows()

