import os,shutil
def split_testdata(args):
    testlist = os.listdir(args.testpath)
    return [testlist[i:i + args.minibatch] for i in range(0, len(testlist), args.minibatch)]
def creat_fold(args):
    if args.model=='test':
        lists=args.split_fold
    elif args.model == 'unsuper':
        lists=[i for i in range(args.cls_num)]
    for lis in lists:
        if not os.path.exists(os.path.join(args.testpath, lis)):
            os.makedirs(os.path.join(args.testpath,lis))
def test_decode(hardcls,softcls,args):
    for filename, cls in zip(args.namelist, hardcls):
        shutil.copy(os.path.join(args.testpath, filename), os.path.join(os.path.join(args.testpath,args.split_fold[cls]), filename))
def unsuper_decode(hardcls,args):
    for filename, cls in zip(args.namelist, hardcls):
        shutil.copy(os.path.join(args.testpath, filename), os.path.join(os.path.join(args.testpath,str(cls)), filename))



