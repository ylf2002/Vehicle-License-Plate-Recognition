import cv2
import numpy as np

image = cv2.imread('pic\car2.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 变成灰度图
# 高斯滤波去噪
blurred = cv2.GaussianBlur(gray, (5, 5), 0, 0, cv2.BORDER_DEFAULT)
# 形态学处理，开运算
kernel = np.ones((23, 23), np.uint8)
opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)  # 开运算
opened = cv2.addWeighted(blurred, 1, opened, -1, 0)

# Otsu大津算法自适应阈值分割
ret, thresh = cv2.threshold(opened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 找到图像边缘
edge = cv2.Canny(thresh, 100, 200)  

# 使用开运算和闭运算让图像边缘连成一个整体
# 形态学处理获取区域，将区域连通，车牌的区域可能在其中
kernel = np.ones((10, 10), np.uint8)
edge1 = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
edge2 = cv2.morphologyEx(edge1, cv2.MORPH_OPEN, kernel)
cv2.imshow('edge',edge2)# 查看边缘图
cv2.imwrite('pic\edge2.jpg',edge2)
# 轮廓
# 查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
contours, hierarchy = cv2.findContours(edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 尝试获取车牌区域
temp_contours = []
for contour in contours:
       if cv2.contourArea(contour) > 500:
              temp_contours.append(contour)
              
car_plates = []
for temp_contour in temp_contours:
       rect_tupple = cv2.minAreaRect(temp_contour)
       rect_width, rect_height = rect_tupple[1]
       if rect_width < rect_height:# 置换，保障长比宽宽
              rect_width, rect_height = rect_height, rect_width
       aspect_ratio = rect_width / rect_height
       # 车牌正常情况下宽高比在2 - 5.5之间
       if aspect_ratio > 2 and aspect_ratio < 5.5:
              car_plates.append(temp_contour)
              rect_vertices = cv2.boxPoints(rect_tupple)
              rect_vertices = np.int0(rect_vertices)
if len(car_plates) == 1:
       for car_plate in car_plates:              
              row_min, col_min = np.min(car_plate[:, 0, :], axis=0)
              row_max, col_max = np.max(car_plate[:, 0, :], axis=0)
              cv2.rectangle(image, (row_min, col_min), (row_max, col_max), (0, 0, 0), 2)
              card = image[col_min:col_max, row_min:row_max ]
              cv2.imshow("img", image)
       cv2.imshow("card_img.jpg", card)
       cv2.imwrite('pic\plate2.jpg',card)


