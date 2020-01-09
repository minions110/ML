import datetime
starttime = datetime.datetime.now()
import numpy as np
from sklearn.model_selection  import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from featureExtraction import base_feature
from utils import normal,util
import os,shutil
import cv2
import argparse
import joblib
import warnings
warnings.filterwarnings("ignore")
def dataload(args):
    print('dataloading-------------------')
    X = []
    Y = []
    namelist=[]
    if args.model=='train':
        for idx, path in enumerate(args.trainpath):
            for file in os.listdir(path):
                image = cv2.imread(os.path.join(path, file))
                hist=base_feature.run(image,args)
                X.append(hist)
                Y.append(idx)
    elif args.model=='test':
        print(len(args.minilist))
        for file in args.minilist:
            if '.jpg'not in file:
                continue
            namelist.append(file)
            Images = cv2.imread(os.path.join(args.testpath, file))
            image = cv2.resize(Images, (256, 256), interpolation=cv2.INTER_CUBIC)
            hist=base_feature.run(image,args)
            X.append(hist)
    if len(args.feature)>1:
        normal.run(args)
    setattr(args, 'X', X)
    setattr(args, 'Y', Y)
    setattr(args, 'namelist', namelist)
def train(args):
    dataload(args)
    X_train, X_test, y_train, y_test = train_test_split(args.X, args.Y,test_size=args.test_size, random_state=1)
    from sklearn.naive_bayes import BernoulliNB
    print('training-------------------')
    clf0 = BernoulliNB().fit(X_train, y_train)
    joblib.dump(clf0, 'weight/' + args.arith+ '_'+args.feature[0] + '_' + args.cls_name + '.pkl')
    print('save weight')
    predictions0 = clf0.predict(X_test)
    print(classification_report(y_test, predictions0))
def test(args):
    fun = joblib.load('weight/' + args.arith + '_' + args.feature[0] + '_' + args.cls_name + '.pkl')
    print('Model loaded successfully')
    for minilist in util.split_testdata(args):
        setattr(args, 'minilist', minilist)
        dataload(args)
        print('predict-------------------')
        hardcls=fun.predict(args.X)
        softcls=fun.predict_proba(args.X)
        # hard_soft=np.vstack((hardcls, softcls)).T
        # np.savetxt('tmp/hard-soft.txt', (hard_soft))
        for filename,cls in zip(args.namelist,hardcls):
            if cls==0:
                shutil.copy(os.path.join(args.testpath,filename),os.path.join(args.savepath[0],filename))
            if cls==1:
                shutil.copy(os.path.join(args.testpath, filename), os.path.join(args.savepath[1], filename))
if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Image ML')
    trainpath=['G:\data\met\Metal bottle','G:\data\met\error']
    testpath='G:\data\met'
    setattr(args, 'trainpath', trainpath)
    setattr(args, 'testpath', testpath)
    setattr(args, 'cls_name', 'MetalBottle')
    setattr(args, 'feature',['HOG'])
    setattr(args, 'normalization','normal_L2')
    setattr(args, 'minibatch', 500)
    setattr(args, 'test_size', 0.2)
    setattr(args, 'arith', 'byes')
    setattr(args, 'model', 'test')
    savepath1 = os.path.join(args.testpath, '1match' + args.feature[0])
    savepath2 = os.path.join(args.testpath, '1nomatch' + args.feature[0])
    if not os.path.exists(savepath1):
        os.mkdir(savepath1)
    if not os.path.exists(savepath2):
        os.mkdir(savepath2)
    setattr(args, 'savepath', [savepath1, savepath2])
    eval(args.model)(args)
    endtime = datetime.datetime.now()
    print(endtime - starttime)



