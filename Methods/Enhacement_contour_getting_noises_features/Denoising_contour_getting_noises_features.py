import cv2
import numpy as np
import glob

for file_name in glob.glob('image*'):
    print(file_name)
    img = cv2.imread(file_name)
    gray_img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2,2) , np.uint8)
    gray_img = cv2.dilate(gray_img , kernel)
    contour , _ = cv2.findContours(gray_img , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE )
    for cnt in contour:
        x , y , w , h = cv2.boundingRect(cnt)
        epsilon = 1.23456789e-14
        approx = cv2.approxPolyDP(cnt,epsilon , True)

        if  w>img.shape[1]*0.95 or h>img.shape[0]*0.95:
            continue

        if cv2.contourArea(cnt) < 35:
            cv2.fillPoly(img, [approx], (255, 255, 255))
            # if cv2.contourArea(cnt)<((h+w)**2*0.05*3.14):
            #     cv2.fillPoly(img, [approx], (255, 0, 0))
            continue

        if  h > w and cv2.contourArea(cnt)<30:
            cv2.fillPoly(img, [approx], (255, 255, 255))
            # if cv2.contourArea(cnt)<((h+w)**2*0.05*3.14):
            #     cv2.fillPoly(img, [approx], (255, 0, 0))
            continue
        # cv2.fillPoly(img,[approx],(0,0,255))

    cv2.imwrite('final_threshed_'+file_name , img)