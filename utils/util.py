import os
def split_testdata(args):
    testlist = os.listdir(args.testpath)
    return [testlist[i:i + args.minibatch] for i in range(0, len(testlist), args.minibatch)]

