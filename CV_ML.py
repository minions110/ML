import datetime
starttime = datetime.datetime.now()
import numpy as np
import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from featureExtraction import base_feature
from Pretreatment import base_pretreat
from arithmetic .base_ML import sklearn_ML
from utils import normal,util
import os,shutil,random
import cv2,time
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
            datalist=os.listdir(path)
            for file in random.sample(datalist,20):
                image = cv2.imread(os.path.join(path, file))
                preimage = base_pretreat.run(image, args)
                hist = base_feature.run(preimage, args)
                X.append(hist)
                Y.append(idx)
    else:
        for file in args.minilist:
            if '.jpg'not in file:
                continue
            namelist.append(file)
            image = cv2.imread(os.path.join(args.testpath, file))
            preimage=base_pretreat.run(image,args)
            hist=base_feature.run(preimage,args)
            X.append(hist)
    setattr(args, 'X', X)
    setattr(args, 'Y', Y)
    setattr(args, 'namelist', namelist)
def train(args):
    dataload(args)
    X_train, X_test, y_train, y_test = train_test_split(args.X, args.Y,test_size=args.test_size, random_state=1)
    print('training-------------------')
    clf=sklearn_ML[args.arith].fit(X_train, y_train)
    joblib.dump(clf, 'weight/' + args.arith+ '_'+args.feature[0] + '_' + args.cls_name + '.pkl')
    print('save weight')
    predictions0 = clf.predict(X_test)
    report=classification_report(y_test, predictions0,output_dict=True)
    df=pd.DataFrame(report).transpose()
    df.to_csv('Report/'+args.arith + '_'+ args.feature[0] + '_' + args.cls_name+
              str(time.strftime("%Y%m%d%H", time.localtime())) +'.txt', sep='\t', index=False)
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
        util.test_decode(hardcls,softcls,args)
def unsuper(args):
    setattr(args, 'minilist', args.testpath)
    dataload(args)
    print('training-------------------')
    clf = sklearn_ML[args.arith].fit(args.X, args.clsnum)
    util.unsuper_decode(clf.labels_,args)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='CV_ML')
    trainpath=['G:/data/ML_samples/metalbottle sample/sample','G:/data/ML_samples/metalbottle sample/neg sample']
    testpath='G:\data\met'
    setattr(args, 'trainpath', trainpath)
    setattr(args, 'testpath', testpath)
    setattr(args, 'cls_name', 'metalbottle')
    setattr(args, 'split_fold', ['match','nomatch'])
    setattr(args, 'pretreat', ['MeanShift'])
    setattr(args, 'feature',['Hog'])
    setattr(args, 'minibatch', 500)
    setattr(args, 'test_size', 0.2)
    setattr(args, 'arith', 'LinearSVC')
    setattr(args, 'cls_num', 2)
    setattr(args, 'model', 'train')
    if args.model != 'train':
        util.creat_fold(args)
    eval(args.model)(args)
    endtime = datetime.datetime.now()
    print(endtime - starttime)



