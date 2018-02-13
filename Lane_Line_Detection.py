import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def img_view_write(img , write = 0 , filename="test.jpg"):
    cv2.imshow("test",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if write == 1 :
        cv2.imwrite(filename,img)

def line_Extraction(img):
    gray_img = cv2.cvtColor(img , cv2.COLOR_RGB2GRAY) #for white only
    img_view_write(gray_img)
    hsv_img  = cv2.cvtColor(img,cv2.COLOR_RGB2HSV) #for colors except white
    img_view_write(hsv_img)

    low_thres_yellow = np.array([25 ,100 , 100] , dtype = 'uint8')
    up_thres_yellow  = np.array([100 ,255 , 255] , dtype = "uint8")
    imBin_yellow = cv2.inRange(hsv_img , low_thres_yellow , up_thres_yellow)

    imBin_white = cv2.inRange(gray_img,220,255)
    ylw_wht_img = cv2.bitwise_or(imBin_white,imBin_yellow)
    gray_ylw_wht = cv2.bitwise_and(gray_img ,ylw_wht_img)
    img_view_write(imBin_white)

    img_g = cv2.GaussianBlur(gray_ylw_wht , (5,5),0)
    #img_view_write(img_g)
    #kernel = np.ones((2,2),np.uint8)
    #img_m = cv2.morphologyEx(gray_ylw_wht,cv2.MORPH_OPEN,kernel)
    #img_view_write(img_m)

    img_cannyEdge = cv2.Canny(img_g , 50 ,150)
    img_view_write(img_cannyEdge)

    h , w = img.shape[:2]
    img_roi = np.copy(img_cannyEdge)
    upper_bound = int(1*h/4)
    lower_bound = int(7*h/8)
    img_roi[0:upper_bound,:] = 0
    img_roi[lower_bound:h,:] = 0
    img_view_write(img_roi)

    rho = 2
    theta = np.pi/180
    min_vote = 40 #20
    minLineLength = 100 #50
    maxLineGap = 5 #180
    lines = cv2.HoughLinesP(img_roi , rho , theta,min_vote,np.array([]),minLineLength,maxLineGap)
    for line in lines :
        for x1,y1,x2,y2 in line:
            img_final = cv2.line(img ,(x1,y1),(x2,y2) ,(255,0,0),2)

    img_view_write(img)

#img = cv2.imread('test/t0.jpg')
img = cv2.imread('test2/solidYellowCurve.jpg')
#img_view_write(img,1,"ok.jpg")
line_Extraction(img)
