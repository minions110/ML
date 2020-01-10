from featureExtraction.colorFeature import *
from featureExtraction.marginFeature import *
import cv2
def HOG(image):
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    hog1 = Hog_descriptor(img.copy(), cell_size=8, bin_size=8)
    vector, hist = hog1.extract()
    return (hist / 255).flatten()
def Hog(image):
    hist = Hogdescriptor(image)
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