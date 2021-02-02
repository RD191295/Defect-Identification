import cv2
import numpy as np
import math
import imutils
from multiprocessing import Process

GOOD_MATCH_PERCENT = 0.7


def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg

Test_Cap = cv2.VideoCapture(0)

Master_Image = cv2.imread("Master_Image.jpg")
while(1):
    _,Test_Image = Test_Cap.read()
    Master_Image_Filter = cv2.bilateralFilter(Master_Image,9,75,75)
    Test_Image_Filter = cv2.bilateralFilter(Test_Image,9,75,75)

    imReg = alignImages(Master_Image_Filter, Test_Image_Filter)
    diff = cv2.absdiff(imReg, Test_Image_Filter)
        
    threshold = 25
    imReg[np.where(diff >  threshold)] = 255
    imReg[np.where(diff <= threshold)] = 0
        
    img_bw = 255*(cv2.cvtColor(imReg, cv2.COLOR_BGR2GRAY) > 15).astype('uint8')
   
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

    thresh = cv2.threshold(mask, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        (x,y, w, h) = cv2.boundingRect(c)
        area= cv2.contourArea(c)
        cv2.rectangle(Test_Image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow('outFilename', Test_Image)
    cv2.waitKey(1)
    
Test_Cap.release()
cv2.destroyAllWindows()
    
    
    
