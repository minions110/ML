from Pretreatment.filtering import *
def run(image,args):
    if args.pretreat==[]:
        return image
    for mod in args.pretreat:
        image = eval(mod)(image)
    return image