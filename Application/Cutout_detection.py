import cv2,os,json
from spectralclustering import *
import multiprocessing
def cutout(imgpath,txtlist,txtpath,savepath):
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    pathlist=[]
    for txtfile in txtlist:
        print(txtfile)
        pointdict={}
        if '.txt' in txtfile:
            image=txtfile[0:-4]+'.jpg'
            if image not in os.listdir(imgpath):
                continue
            img=cv2.imread(os.path.join(imgpath,image))
            h,w,_=img.shape
            file=open(os.path.join(txtpath,txtfile),'r')
            for idx,line in enumerate(file):
                line=line.replace('\n','').split(' ')
                if line[0] not in pointdict:
                    pointdict[line[0]]=[]
                if len(line)==5:
                    y1= 0 if int(line[2])<0 else int(line[2])
                    y2=h if int(line[4])>h else int(line[4])
                    x1 =  0 if int(line[1])<0 else int(line[1])
                    x2 = w if int(line[3])>w else int(line[3])
                elif len(line)==6:
                    y1 = 0 if int(line[3])<0 else int(line[3])
                    y2 = h if int(line[5])>h else int(line[5])
                    x1 = 0 if int(line[2])<0 else int(line[2])
                    x2 = w if int(line[4])>w else int(line[4])
                else:
                    raise Exception("File {}:The row elements are {}".format(image, len(line)))
                img1 = img[y1:y2, x1:x2]
                save = os.path.join(savepath, line[0][0:4])
                if not os.path.exists(save):
                    os.mkdir(save)
                    pathlist.append(save)
                cv2.imwrite(os.path.join(save, txtfile[0:-4] + '_'+str(idx).zfill(2) + '.jpg'), img1)
        elif '.json' in txtfile:
            image=txtfile[0:-5]+'.jpg'
            if image not in os.listdir(imgpath):
                continue
            img=cv2.imread(os.path.join(imgpath,image))
            file = open(os.path.join(txtpath, txtfile), 'r')
            alldata=json.load(file)
            for idx,shapes in enumerate(alldata['shapes']):
                X,Y=[],[]
                for point in shapes['points']:
                    X.append(point[0])
                    Y.append(point[1])
                img1 = img[min(Y):max(Y), min(X):max(X)]
                save = os.path.join(savepath, shapes['label'][0:4])
                if not os.path.exists(save):
                    os.mkdir(save)
                    pathlist.append(save)
                cv2.imwrite(os.path.join(save, txtfile[0:-4] + '_'+str(idx).zfill(2) + '.jpg'), img1)
    return pathlist
def getsplit(ii, nu):
    """
    ii : the list of input.
    nu : the number of parts you will split into .
    """
    Nlen = len(ii)
    res = []
    if Nlen%nu == 0:
        nper = int(Nlen/nu)
    else:
        nper = int(Nlen/nu) + 1
    for i in range(0, Nlen, nper):
        res.append(ii[i:i+nper])
    return res
if __name__ == '__main__':
    imgpath='E:\dataset/1807data'
    txtpath='E:\dataset/1cls_lip\m_v3_5.9_resnet50_lpf_epoch30_010309/test_result'
    savepath='E:\dataset/1cls_lip\m_v3_5.9_resnet50_lpf_epoch30_010309/cutout1'
    txtlist=os.listdir(txtpath)
    numofProcess = 1
    multi_processing_datas = getsplit(txtlist, numofProcess)
    plist = []
    for pdata in multi_processing_datas:
        p = multiprocessing.Process(target=cutout,
                                    args=(imgpath,pdata,txtpath,savepath))
        plist.append(p)

    for pro in plist:
        pro.start()
    for pro in plist:
        pro.join()

