from sklearn.preprocessing import Normalizer
def normal_L1(args):
    norm2 = Normalizer(norm='l1')
    args.X = norm2.fit_transform(args.X)
def normal_L2(args):
    norm2 = Normalizer(norm='l2')
    args.X = norm2.fit_transform(args.X)
def run(arge):
    eval(arge.normalization)(arge)

