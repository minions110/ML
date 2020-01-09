import cv2
import numpy as np
from scipy import signal
import numpy as np
import cv2
def Ideal_Hfiltering(image,D0=30):
    (r,g,b) = cv2.split(image)
    image = cv2.merge([b,g,r])
    J = np.double(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY))
    Fuv = np.fft.fftshift(np.fft.fft2(J))
    m,n = image.shape[0],image.shape[1]
    xo = np.floor(m/2)
    yo = np.floor(n/2)
    h = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            D = np.sqrt((i-xo)**2+(j-yo)**2)
            if D>=D0:
                h[i,j]=1
            else:
                h[i,j]=0
    Guv1 = h*Fuv
    g = np.fft.ifftshift(Guv1)
    return np.uint8(np.real(np.fft.ifft2(g)))

def butterworth_low_pass_kernel(img,cut_off,butterworth_order=1):
    r,c = img.shape[1],img.shape[0]
    u = np.arange(r)
    v = np.arange(c)
    u, v = np.meshgrid(u, v)
    low_pass = np.sqrt( (u-r/2)**2 + (v-c/2)**2 )
    denom = 1.0 + (low_pass / cut_off)**(2 * butterworth_order)
    low_pass = 1.0 / denom
    return low_pass

def butterworth_high_pass_kernel(img,D0=5,n=1):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 1 - butterworth_low_pass_kernel(img,D0,n)
    gray = np.float64(img)
    gray_fft = np.fft.fft2(gray)
    gray_fftshift = np.fft.fftshift(gray_fft)
    dst_filtered = kernel * gray_fftshift
    dst_ifftshift = np.fft.ifftshift(dst_filtered)
    dst_ifft = np.fft.ifft2(dst_ifftshift)
    dst = np.abs(np.real(dst_ifft))
    dst = np.clip(dst,0,255)
    return np.uint8(dst)
def filter2D(image):
    gaussian =cv2.getGaussianKernel(5,10)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray=cv2.filter2D(gray,-1,gaussian)
    return gray
def sepFilter2D(image):
    M = np.array([-1, 2, -1])
    G = cv2.getGaussianKernel(ksize=3, sigma=-1)
    Lx = cv2.sepFilter2D(src=image, ddepth=cv2.CV_64F, kernelX=M, kernelY=G)
    Ly = cv2.sepFilter2D(src=image, ddepth=cv2.CV_64F, kernelX=G, kernelY=M)
    FM = np.abs(Lx) + np.abs(Ly)
    return cv2.mean(FM)[0]
def blur(image):
    return cv2.blur(image, (3, 3))
def GaussianBlur(image):
    return cv2.GaussianBlur(image, (7,7), 0)
def medianBlur(image):
    return cv2.medianBlur(image, 7)
def boxFilter(image):
    return cv2.boxFilter(image, -1, (2, 2), normalize=1)
def bilateralFilter(image):
    return cv2.bilateralFilter(image,9,75,75)

if __name__ == "__main__":
    img=cv2.imread('G:/data/met/matchHOG/image_180704150028_6693_View1_00.jpg')
    img1=Ideal_Hfiltering(img.copy(),D0=30)
    img2=filter2D(img.copy())
    img3=butterworth_high_pass_kernel(img.copy(),D0=5)
    cv2.imshow('img1',img1)
    cv2.imshow('img2', img2)
    cv2.imshow('img3', img3)
    cv2.waitKey()

'''
安装opencv主模块和contrib附加模块步骤：
pip uninstall opencv-python （如果已经安装opencv-python包，先卸载）
pip install opencv-contrib-python
'''
# cv2.xmingproc.jointBilateralFilter(joint,src,d,sigmaColor,sigmaSpace,borderType)
# cv2.ximgproc.guidedFilter(guide,src,radius,eps,dDepth)
# signal.convolve2d(img,kernel,mode,boundary,fillvalue)
# cv2.Sobel()
# cv2.Schar()
# cv2.Laplacian()