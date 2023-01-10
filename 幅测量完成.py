import cv2
import numpy as np
import time
import imutils
import matplotlib.pyplot as plt
import pandas as pd


#カメラの設定　デバイスIDは0
cap = cv2.VideoCapture(0)
img = cv2.imread("4f7fd727534f1f977a27836834564fc.png", 0)
pixel = 0.06 #mm
x = 180
y = 200
w = 300
h = 300
xmin,xmax = 0,600
ymin,ymax = 200,200
bgrLower = np.array([80,200, 200])    # 抽出する色の下限(BGR)
bgrUpper = np.array([255,255, 255])
#kernel1 = np.ones((3,3),np.uint8) 
kernel2 = np.array([[0,1,0],
       [0,1,0],
       [0,1,0],
       [0,1,0],
       [0,1,0],
       [1,1,1],
       [0,1,0],
       [0,1,0],
       [0,1,0],
       [0,1,0],
       [0,1,0]], np.uint8)

#繰り返しのためのwhile文
while True:
    #カメラからの画像取得
    ret, frame = cap.read()

    frame = cv2.resize(frame,(800,800))
    img_mask = cv2.inRange(frame, bgrLower, bgrUpper)#二値化 
    #crop_img = img_mask[y:y+h,x:x+w]#画面キャスト

    # 収縮処理
    erosion = cv2.erode(img_mask,kernel2,iterations = 1) #糸芯抽出
    #gradient = cv2.morphologyEx(img_mask, cv2.MORPH_GRADIENT, kernel1) 

    imgGray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#グレースケール
    imgBLUR = cv2.GaussianBlur(imgGray,(5,5),1)#ガウシアンぼかし
    imgCanny = cv2.Canny(imgBLUR,120,120)#糸輪郭

    cnts = cv2.findContours(erosion.copy(), cv2.RETR_EXTERNAL,
	    cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    #計測
    analysis_row=200
    test_line=erosion[analysis_row,:]#erosionの２００行目のすべてのピクセル
    test_line2=img_mask[analysis_row,:]
    num = np.count_nonzero(test_line) #糸直径ピクセル数
    length = round(num*pixel,2) #直径算出,小数点以下1桁
    coleft = np.nonzero(test_line2)
    right = np.amax(coleft)
    left = np.amin(coleft)
    center = int((right+left)/2)

   #cv2.circle(erosion, (left,200), 8, (255, 0, 0), -1)
    #cv2.circle(erosion, (right,200), 8, (255, 0, 0), -1)
    cv2.arrowedLine(erosion,(left-80,200),(left,200),(255,255,255),thickness=1,line_type=cv2.LINE_4)
    cv2.arrowedLine(erosion,(right+80,200),(right,200),(255,255,255),thickness=1,line_type=cv2.LINE_4)
    cv2.circle(frame,(center,200),6,(0,0,255),-1)
   # cv2.line(erosion, (30, 60), (600, 60), (0, 100, 100), thickness=5, lineType=cv2.LINE_4)

    cv2.putText(erosion,'yarn diameter = ' + str(length)+'mm', (left+100,200),cv2.FONT_HERSHEY_PLAIN,1,(255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame,'yarn diameter = ' + str(length)+'mm', (0,20),cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0), 2, cv2.LINE_AA)
    #カメラの画像の出力

    print(length)
    #time.sleep(0.1)

    cv2.imshow("origin",frame)
    cv2.imshow("erosion",erosion)
    cv2.imshow('Canny',imgCanny)
    #print(left)
    #print(right)

    #繰り返し分から抜けるためのif文
    key =cv2.waitKey(200)
    if key == 27:
        break

#メモリを解放して終了するためのコマンド
cap.release()
cv2.destroyAllWindows()