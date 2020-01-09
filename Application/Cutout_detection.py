import cv2,os,json
from spectralclustering import *
import multiprocessing
def cutout(imgpath,txtlist,txtpath,savepath):
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    pathlist=[]
    for txtfile in txtlist:
        pointdict={}
        if '.txt' in txtfile:
            image=txtfile[0:-4]+'.jpg'
            if image not in os.listdir(imgpath):
                continue
            img=cv2.imread(os.path.join(imgpath,image))
            file=open(os.path.join(txtpath,txtfile),'r')
            for idx,line in enumerate(file):
                line=line.replace('\n','').split(' ')
                if line[0] not in pointdict:
                    pointdict[line[0]]=[]
                if len(line)==5:
                    img1=img[int(line[2]):int(line[4]),int(line[1]):int(line[3])]
                if len(line)==6:
                    img1 = img[int(line[3]):int(line[5]), int(line[2]):int(line[4])]
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
    imgpath='C:/Users/zkteco/Desktop/test/img'
    txtpath='C:/Users/zkteco/Desktop/test/test_result'
    savepath='C:/Users/zkteco/Desktop/test/cutout'
    txtlist=os.listdir(txtpath)
    numofProcess = 20
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

