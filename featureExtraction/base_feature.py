from featureExtraction.colorFeature import *
from featureExtraction.marginFeature import *
import cv2
def Hog(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = Hog_descriptor(img)
    return (hist / 255).flatten()
def Lbp(image):
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    hist=Lbp_descriptor(img)
    return (hist / 255).flatten()
def histogram(image):
    hist=color_histogram(image)
    return (hist / 255).flatten()
def moments(image):
    color_feature=color_moments(image)
    return np.array(color_feature) / 100
def run(image,args):
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
    for mod in args.feature:
        hist=eval(mod)(image)
    return hist