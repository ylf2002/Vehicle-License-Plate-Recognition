#######################################
#    School of Software Technology    #
#   Dalian University of Technology   #
#             yang lifan              #
#          2862506026@qq.com          #
#######################################

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('pic\car2.jpg')
'''*————————*提取出车牌*————————*'''

# 颜色空间转换
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# 蓝色的范围，不同环境下不一样，可灵活调整
lower_blue = np.array([100, 80, 90])
upper_blue = np.array([120, 255, 255])
license_plate = cv2.inRange(hsv, lower_blue, upper_blue)
onlylic_plate = cv2.bitwise_and(img, img, mask = license_plate)

# 转换为灰度图
img_gray = cv2.cvtColor(onlylic_plate, cv2.COLOR_BGR2GRAY)
# 自适应高斯阈值分割
thresh = cv2.adaptiveThreshold(
    img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 6)
cv2.imshow('thresh.jpg',thresh)
# 边缘检测
canny = cv2.Canny(thresh, 100, 200)
kernel1 = np.ones((12,40), np.uint8)  
img_edge1 = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel1)         # 闭运算  
img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel1)  # 开运算
cv2.imshow('img_edge1.jpg',img_edge1)
cv2.imshow('img_edge2.jpg',img_edge2)
# cv2.imwrite('pic\edge5.jpg',img_edge2)
# 轮廓
contours ,hierarchy = cv2.findContours(img_edge2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# 通过长宽比判断车牌的位置，并截取
for contour in contours:
       rect = cv2.minAreaRect(contour)   #找出最小外接矩形 中心点、宽和高、角度
       if rect[1][1]>rect[1][0]:            
            k=rect[1][1]/rect[1][0]
       else:
            k=rect[1][0]/rect[1][1]
       if (k>2.5)&(k<5):    #判断车牌的轮廓
        
            a=cv2.boxPoints(rect)     #获取外接矩形的四个点
            box = np.int0(a)
            aa=cv2.drawContours(img, [box], 0, (0, 0, 0), 1)  #找出车牌的位置（不要颜色）     
            x=[]
            y=[]
            for i in range(4):
                x.append(box[i][1])
                y.append(box[i][0])
            min_x=min(x)
            max_x=max(x)
            min_y=min(y)
            max_y=max(y)         
            cut=img[min_x:max_x,min_y:max_y]
            # cv2.imwrite('pic\plate5.jpg',cut)  
            cv2.imshow('plate4.jpg',cut)
# 对截取到的车牌进行处理        
