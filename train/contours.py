import cv2,os
import copy
import numpy as np
from Application.houghlines import *
from Application.balances import white_balance_5
imgpath='G:\data\met'
imglist=os.listdir(imgpath)
for image in imglist:
    if '.jpg' in image:
        img = cv2.imread(os.path.join(imgpath,image))
        img1=img.copy()
        img2=img.copy()
        img1=line_detect_possible_demo(img1)
        # img2 = line_detection(img2)
        cv2.imshow("img1", img1)
        # cv2.imshow("img2", img2)

        cv2.waitKey()
        # kernel = np.ones((3, 3))
        # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # # binary = cv2.Canny(gray, 50, 150)
        #
        # contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #
        # draw_img = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 3)
        #
        # print("contours:类型：", type(contours))
        # print("第0 个contours:", type(contours[0]))
        # print("contours 数量：", len(contours))
        #
        # print("contours[0]点的个数：", len(contours[0]))
        # print("contours[1]点的个数：", len(contours[1]))
        #
        # cv2.imshow("img", img)
        # cv2.imshow("draw_img", draw_img)
        #
        # cv2.waitKey()
        # allarea = np.concatenate(contours)
        # for contour in contours:
        # # 合并所有轮廓点集合
        #     rect = cv2.minAreaRect(contour)
        #     box = cv2.boxPoints(rect)
        #     box = np.int0(box)
        #     # 画出来
        #
        #     boximg=cv2.drawContours(img.copy(), [box], 0, (255, 0, 0), 1)
        #     cv2.imshow("boximg", boximg)
        #     cv2.waitKey()

        # #白平衡
        # img=white_balance_5(img)
        # #形态学滤波
        # kernel = np.ones((3, 3))
        # img_kn = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        # img_kn = cv2.morphologyEx(img_kn, cv2.MORPH_CLOSE, kernel)
        # #灰度图
        # img_gray=cv2.cvtColor(img_kn.copy(), cv2.COLOR_BGR2GRAY)
        # #边缘检测
        # Cannyimg=cv2.Canny(img_gray,50,150)
        # #获取轮廓
        # contours, hier = cv2.findContours(Cannyimg,  cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        # for contour in contours:
        #     # print(contour)
        #     are=cv2.contourArea(contour)
        #     if are<100:
        #         continue
        #     else:
        #         cv2.drawContours(img, contour, -1, (0, 0, 255), 2)
        #     cv2.imshow("contours", img)
        #     cv2.waitKey()
        #     cv2.destroyAllWindows()
