# -*- coding: utf-8 -*-
from PCV.tools import imtools, pca
from PIL import Image, ImageDraw
from pylab import *
from scipy.cluster.vq import *
import os
import warnings
warnings.filterwarnings("ignore")
import shutil

def run(path,k):
    imlist = imtools.get_imlist(path)
    imnbr = len(imlist)
    print(imnbr)
    image_array = []
    for im in imlist:
        img = Image.open(im)
        if img.size[0]< img.size[1]:
            image_array.append(array(img.resize((80,100))).flatten())
        else:
            img = img.rotate(90)
            image_array.append(array(img.resize((80,100))).flatten())


    # Load images, run PCA.
    immatrix = array([array(Image.open(im).resize((200,200))).flatten() for im in imlist], 'f')
    V, S, immean = pca.pca(immatrix)
    print(len(V))
    print(len(V[0]))
    # Project on 2 PCs.
    projected = array([dot(V[[0, 1]], immatrix[i] - immean) for i in range(imnbr)])  # P131 Fig6-3左图
    # projected = array([dot(V[[1, 2]], immatrix[i] - immean) for i in range(imnbr)])  # P131 Fig6-3右图

    n = len(projected)
    # compute distance matrix
    S = array([[sqrt(sum((projected[i] - projected[j]) ** 2))
                for i in range(n)] for j in range(n)], 'f')
    # create Laplacian matrix
    rowsum = sum(S, axis=0)
    D = diag(1 / sqrt(rowsum))
    I = identity(n)
    L = I - dot(D, dot(S, D))
    # compute eigenvectors of L
    U, sigma, V = linalg.svd(L)
    # create feature vector from k first eigenvectors
    # by stacking eigenvectors as columns
    features = array(V[:k]).T
    # k-means
    features = whiten(features)
    centroids, distortion = kmeans(features, k)
    code, distance = vq(features, centroids)
    # plot clusters
    print(code)

    count = 0
    for c in range(k):
        count += 1
        elements = where(code == c)[0]
        nbr_elements = len(elements)
        os.makedirs(path + '/result/'+ str(count))
        for p in range(nbr_elements):
            file_path, temp_filename = os.path.split(imlist[elements[p]])
            file_name, exten = os.path.splitext(temp_filename)
            src = imlist[elements[p]]
            dst = path + '/result/'+ str(count) + '/' + temp_filename
            shutil.copyfile(src, dst)

    for c in range(k):
        ind = where(code == c)[0]
        figure()
        gray()
        for i in range(minimum(len(ind), 39)):
            im = Image.open(imlist[ind[i]])
            subplot(4, 10, i + 1)
            imshow(array(im))
            axis('equal')
            axis('off')
    show()
if __name__ == '__main__':
    path = 'C:/Users/zkteco/Desktop/test/cutout/Liqu/result/1'
    k=2
    run(path,k)