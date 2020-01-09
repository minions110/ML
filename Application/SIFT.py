# #python
# #!/usr/bin/env python
# #-*- coding:utf-8 -*-
# from PIL import Image
# from pylab import *
# from PCV.localdescriptors import sift
# from PCV.localdescriptors import harris
# import os
# root='C:/Users/zkteco/Desktop/test/cutout/Liqu/'
# imname = ('empire.jpg')
# siftname = ('empire.sift')
# im = array(Image.open(root+imname).convert('L'))
# sift.process_image(root+imname,root+siftname)
# l1,d1= sift.read_features_from_file(root+siftname)
# figure()
# gray()
# subplot(131)
# #图1 ：SIFT特征
# sift.plot_features(im,l1,circle = False)
# title('sift-features')
# subplot(132)
# #图2 ：使用圆圈表示特征尺度的SIFT特征
# sift.plot_features(im,l1,circle = True)
# title('sift_features_det')
# harrisim = harris.compute_harris_response(im)
# filtered_coords = harris.get_harris_points(harrisim,6,0.1)
# subplot(133)
# """
# 图3 ：harris角点检测的结果
# """
# imshow(im)
# plot([p[1]for p in filtered_coords],[p[0] for p in filtered_coords])
# axis('off')
# title('harris')
# show()

from PIL import Image
from pylab import *
from PCV.localdescriptors import sift


im1f='C:/Users/zkteco/Desktop/test/cutout/Liqu/image_180704071058_607_View1_02.jpg'
im2f='C:/Users/zkteco/Desktop/test/cutout/Liqu/image_180704070735_565_View1_01.jpg'
im1=array(Image.open(im1f))
im2=array(Image.open(im2f))

sift.process_image(im1f, '01.sift')
l1, d1 = sift.read_features_from_file('01.sift')
figure()
gray()
subplot(121)
sift.plot_features(im1,l1,circle=False)

sift.process_image(im2f,'out_sift_2.txt')
l2,d2=sift.read_features_from_file('out_sift_2.txt')
subplot(122)
sift.plot_features(im2,l2,circle=False)

matches=sift.match_twosided(d1,d2)
print('{}matches'.format(len(matches.nonzero()[0])))
figure()
gray()
sift.plot_matches(im1,im2,l1,l2,matches,show_below=True)
show()