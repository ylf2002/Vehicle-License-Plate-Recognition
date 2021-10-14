import cv2
 
img = cv2.imread("E:\\vehicle_license_plate_recognition\\yolov5-plate\\runs\\detect\\exp\\crops\\licence\\car0.jpg")  # 读取图片
# cv2.imshow("img",img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 转换了灰度化
 
# 2、将灰度图像二值化
img_thre = cv2.adaptiveThreshold(
    img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 6)
 
# 3、分割字符
white = []  # 记录每一列的白色像素总和
black = []  # ..........黑色.......
height = img_thre.shape[0]
width = img_thre.shape[1]
white_max = 0
black_max = 0
# 计算每一列的黑白色像素总和
for i in range(width):
    s = 0  # 这一列白色总数
    t = 0  # 这一列黑色总数
    for j in range(height):
        if img_thre[j][i] == 255:
            s += 1
        if img_thre[j][i] == 0:
            t += 1
    white_max = max(white_max, s)
    black_max = max(black_max, t)
    white.append(s)
    black.append(t)
 
j = True  # False表示白底黑字；True表示黑底白字
if black_max > white_max:
    j = True
 
# 分割图像
def find_end(start_):
    end_ = start_+1
    for m in range(start_+1, width-1):
        if (black[m] if j else white[m]) > (0.85 * black_max if j else 0.9 * white_max):  
            end_ = m
            break
    return end_
 
n = 1
start = 1
end = 2
while n < width-2:
    n += 1
    if (white[n] if j else black[n]) > (0.25 * white_max if j else 0.1 * black_max):
        start = n
        end = find_end(start)
        n = end
        if end-start > 5:
            cj = img_thre[1:height, start:end]
            #cv2.imwrite('pic/char/' + str(n)+".bmp", cj)
            
            #cv2.imshow('cj.jpg',cj)
            #cv2.waitKey(0)

print("end!")
