import numpy as np
import cv2

print "Shape Matching Using Fourier Descriptor"

distThreshold = 0.08
ix, iy = -1, -1
rect = (0, 0, 1, 1)

manually = False
temSeleteFlag = False
temReadyFlag = False
temConfirmFlag = False
matchOverFlag = False

templeteComVector = []
sampleComVectors = []
sampleContours = []

# Manually select templete by mouse, On/Off by manually flag
def selectTemplete(event, x, y, flags, param):
    global rect, temSeleteFlag, temReadyFlag, ix, iy

    if event == cv2.EVENT_LBUTTONDOWN and temReadyFlag == False:
        temSeleteFlag = True
        ix, iy = x, y
       
    elif event == cv2.EVENT_LBUTTONUP:
        if temReadyFlag == False and temSeleteFlag == True:
            # rect is selected templete ROI
            rect = (min(ix,x), min(iy,y), abs(ix-x), abs(iy-y))
            # draw a blue rectangle after selection
            cv2.rectangle(imgOri, (ix,iy), (x,y), (255,0,0), 2)
        temSeleteFlag = False
        temReadyFlag = True

# Main findcontour function 
def getContours(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold white paper(background) to white pixel(255), word is actully black(0)
    retvalth, imgthreshold = cv2.threshold(imgray, 50, 255, cv2.THRESH_BINARY)
    # We want words are white, backgournd is black, easy for opencv findcontour function
    imgthresholdNot = cv2.bitwise_not(imgthreshold)
    # Dilation make all 6 to form a closed loop
    kernel = np.ones((5,5), np.uint8)
    imgdilation = cv2.dilate(imgthresholdNot, kernel, iterations=2)
    # Must use EXTERNAL outer contours, Must use CHAIN_APPROX_NONE method(not change points)
    imgcontours, contours, hierarchy = cv2.findContours(imgdilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    return contours

# Get complex vector of templete contour
def getTempleteCV():
    # This is the templete region that we select by mouse or default
    templeteROI = imgOricpy[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
    # Automatically find templete contour
    tpContour = getContours(templeteROI)

    for contour in tpContour:
        x, y, w, h = cv2.boundingRect(contour)
        for point in contour:
            # -x and -y are to make left and upper boundry start from 0
            templeteComVector.append( complex(point[0][0]-x, (point[0][1]-y)))

# Get complex vectors of testees contours
def getSampleCV():
    spContours = getContours(imgOricpy)       
    # cv2.drawContours(imgOri, spContours, -1, (0, 0, 255), 1)
    
    for contour in spContours:
        sampleComVector = []
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(imgOri, (x,y), (x+w,y+h), (100,100,100), 1)

        for point in contour:
            sampleComVector.append( complex(point[0][0]-x, (point[0][1]-y)) )
        # sampleComVectors store CV of all testees contours 
        sampleComVectors.append(sampleComVector)
        # sampleContours store all testees contours, same order with sampleComVectors
        sampleContours.append(contour)

# Calculate fourier transform of templete CV
def getempleteFD():
    
    return np.fft.fft(templeteComVector)

# Calculate fourier transform of sample CVs
def getsampleFDs():
    FDs = []
    for sampleVector in sampleComVectors:
        sampleFD = np.fft.fft(sampleVector)
        FDs.append(sampleFD)

    return FDs    

# Make fourier descriptor invariant to rotaition and start point
def rotataionInvariant(fourierDesc):
    for index, value in enumerate(fourierDesc):
        fourierDesc[index] = np.absolute(value)

    return fourierDesc    

# Make fourier descriptor invariant to scale
def scaleInvariant(fourierDesc):
    firstVal = fourierDesc[0]

    for index, value in enumerate(fourierDesc):
        fourierDesc[index] = value / firstVal

    return fourierDesc

# Make fourier descriptor invariant to translation
def transInvariant(fourierDesc):
    
    return fourierDesc[1:len(fourierDesc)]

# Get the lowest X of frequency values from the fourier values.
def getLowFreqFDs(fourierDesc):
    # frequence order returned by np.fft is (0, 0.1, 0.2, 0.3, ...... , -0.3, -0.2, -0.1)
    # Note: in transInvariant(), we already remove first FD(0 frequency)

    return fourierDesc[:5]

# Get the final FD that we want to use to calculate distance
def finalFD(fourierDesc):
    fourierDesc = rotataionInvariant(fourierDesc)
    fourierDesc = scaleInvariant(fourierDesc)
    fourierDesc = transInvariant(fourierDesc)
    fourierDesc = getLowFreqFDs(fourierDesc)

    return fourierDesc

# Core match function
def match(tpFD, spFDs):
    tpFD = finalFD(tpFD)
    # dist store the distance, same order as spContours
    dist = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for spFD in spFDs:
        spFD = finalFD(spFD)
        # Calculate Euclidean distance between templete and testee
        dist.append( np.linalg.norm(np.array(spFD)-np.array(tpFD)) )
        x, y, w, h = cv2.boundingRect(sampleContours[len(dist)-1])
        # Draw distance on image
        distText = str(round(dist[len(dist)-1],2))
        cv2.putText(imgOri,distText,(x,y-8), font,0.5,(0,0,0),1,cv2.LINE_AA)
        # print str(len(dist)) + ": " + str(dist[len(dist)-1])
        # if distance is less than threshold, it will be good match.
        if dist[len(dist)-1] < distThreshold:   
            cv2.rectangle(imgOri, (x-5,y-5), (x+w+5,y+h+5), (40,255,0), 2)

# -------------------------------------------------------------------------- 
# Main loop
imgOri = cv2.imread("a2.bmp", 1)
# imOricpy is for processing, imgOri is for showing
imgOricpy = imgOri.copy()
cv2.namedWindow("Original Image")

if manually == True:
    # Manually select templete by mouse
    cv2.setMouseCallback("Original Image", selectTemplete)
else:
    # Default region: upper 6
    rect = (50, 100, 130, 160)
    cv2.rectangle(imgOri, (50, 100), (180,260), (255,0,0), 2)
    temReadyFlag = True
    temConfirmFlag = True   
    
while(True):
    
    cv2.imshow("Original Image", imgOri)
    
    if temReadyFlag == True and matchOverFlag == False and temConfirmFlag == True:
        # Get complex vector
        getTempleteCV()
        getSampleCV()
        # Get fourider descriptor
        tpFD = getempleteFD()
        sampleFDs = getsampleFDs()
        # real match function
        match(tpFD, sampleFDs)
        
        matchOverFlag = True
        cv2.imwrite("result.jpg", imgOri)
        # Resize img for showing
        imgShow = cv2.resize(imgOri, None, fx=0.66, fy=0.66, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Small Size Show", imgShow)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('y') or key == ord('Y'):
        # Press Y for templete confirm once mouse selection done
        temConfirmFlag = True
    elif key == ord('q') or key == ord('Q'):
        # Press q for quit
        break
 
cv2.destroyAllWindows()
