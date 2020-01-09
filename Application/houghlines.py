import cv2
import numpy as np
def line_detect_possible_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # apertureSize是sobel算子大小，只能为1,3,5，7
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 10, minLineLength=60,maxLineGap=10)  #函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
    return image

def line_detection(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize=3)    #apertureSize是sobel算子大小，只能为1,3,5，7
    lines = cv2.HoughLines(edges,1,np.pi/360,200)  #函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
    return image