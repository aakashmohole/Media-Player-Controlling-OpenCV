import cv2
import numpy as np 
import math as m
import pyautogui as p
import time as t

#Read Camera
cap = cv2.VideoCapture(0)

# incresing brightness of camera
cap.set(10,200)

def nothing(x):
    pass

#window name
cv2.namedWindow("Threshold Adjustments",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Threshold Adjustments", (300, 300)) 
cv2.createTrackbar("Thresh", "Threshold Adjustments", 0, 255, nothing)

#COlor Detection Track

cv2.createTrackbar("Lower_H", "Threshold Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower_S", "Threshold Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower_V", "Threshold Adjustments", 0, 255, nothing)
cv2.createTrackbar("Upper_H", "Threshold Adjustments", 255, 255, nothing)
cv2.createTrackbar("Upper_S", "Threshold Adjustments", 255, 255, nothing)
cv2.createTrackbar("Upper_V", "Threshold Adjustments", 255, 255, nothing)


while True:
    _,frame = cap.read()
    frame = cv2.flip(frame,2)
    frame = cv2.resize(frame,(600,500))
    # Get hand data from the rectangle sub window
    cv2.rectangle(frame, (0,1), (300,500), (255, 0, 0), 0)
    img_crop = frame[1:500, 0:300]
    
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    
        
    #detecting hand
    lower_h = cv2.getTrackbarPos("Lower_H", "Threshold Adjustments")
    lower_s = cv2.getTrackbarPos("Lower_S", "Threshold Adjustments")
    lower_v = cv2.getTrackbarPos("Lower_V", "Threshold Adjustments")

    upper_h = cv2.getTrackbarPos("Upper_H", "Threshold Adjustments")
    upper_s = cv2.getTrackbarPos("Upper_S", "Threshold Adjustments")
    upper_v = cv2.getTrackbarPos("Upper_V", "Threshold Adjustments")

    lower_bound = np.array([lower_h, lower_s, lower_v])
    upper_bound = np.array([upper_h, upper_s, upper_v])

    #Creating Mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    #filter mask with image
    filtr = cv2.bitwise_and(img_crop, img_crop, mask=mask)
    
    mask1  = cv2.bitwise_not(mask)
    m_g = cv2.getTrackbarPos("Thresh", "Threshold Adjustments") #getting track bar value
    ret,thresh = cv2.threshold(mask1,m_g,255,cv2.THRESH_BINARY)
    dilata = cv2.dilate(thresh,(3,3),iterations = 6)
    
    #findcontour(img,contour_retrival_mode,method)
    cnts,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    try:
        # Find contour with maximum area
        cm = max(cnts, key=lambda x: cv2.contourArea(x))
        #print("C==",cnts)
        epsilon = 0.0005*cv2.arcLength(cm,True)
        data= cv2.approxPolyDP(cm,epsilon,True)
        
        hull = cv2.convexHull(cm)
        
        cv2.drawContours(img_crop, [cm], -1, (50, 50, 150), 2)
        cv2.drawContours(img_crop, [hull], -1, (0, 255, 0), 2)

        # Find convexity defects
        hull = cv2.convexHull(cm, returnPoints=False)
        defects = cv2.convexityDefects(cm, hull)
        count_defects = 0
        
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
           
            start = tuple(cm[s][0])
            end = tuple(cm[e][0])
            far = tuple(cm[f][0])
            #Cosin Rule
            a = m.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = m.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = m.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (m.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            if angle <= 50:
                count_defects += 1
                cv2.circle(img_crop,far,5,[255,255,255],-1)
        
        print("count==",count_defects)
        
        # Print number of fingers
        if count_defects == 0:
            
            cv2.putText(frame, " ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
        elif count_defects == 1:
            
            p.press("space")
            cv2.putText(frame, "Play/Pause", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
        elif count_defects == 2:
            p.press("up")
            
            cv2.putText(frame, "Volume UP", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
        elif count_defects == 3:
            p.press("down")
            
            cv2.putText(frame, "Volume Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
        elif count_defects == 4:
            p.press("right")
            
            cv2.putText(frame, "Forward", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
        else:
            pass
           
    except:
        pass
    #step -10    
    cv2.imshow("Thresh", thresh)
    #cv2.imshow("mask==",mask)
    cv2.imshow("filter==",filtr)
    cv2.imshow("Result", frame)

    key = cv2.waitKey(25) &0xFF    
    if key == 27: 
        break
cap.release()
cv2.destroyAllWindows()