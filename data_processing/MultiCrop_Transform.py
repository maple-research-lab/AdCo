#modified from https://github.com/facebookresearch/swav/blob/master/src/multicropdataset.py
from torchvision import transforms
from moco.loader import GaussianBlur
from Data_Processing.Finetune_Augment import RandAugment
class Multi_Fixtransform(object):
    def __init__(
            self,
            nmb_crops,
            size_crops,

            min_scale_crops,
            max_scale_crops,normalize):

        assert len(min_scale_crops) == len(size_crops)
        assert len(max_scale_crops) == len(size_crops)
        trans=[]


        for i in range(len(size_crops)):
            repeat_times=nmb_crops[i]
            for k in range(repeat_times):
                randomresizedcrop = transforms.RandomResizedCrop(
                    size_crops[i],
                    scale=(min_scale_crops[i], max_scale_crops[i]),
                )
                weak = transforms.Compose([
                    randomresizedcrop,
                    # transforms.RandomApply([
                    #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    # ], p=0.8),
                    # transforms.RandomGrayscale(p=0.2),
                    # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])
                trans.append(weak)
        self.trans=trans
        print("In total we have %d transformations"%len(self.trans))
    def __call__(self, x):
        multi_crops = list(map(lambda trans: trans(x), self.trans))
        return multi_crops
from moco.RandAugmentMC import RandAugmentMC
class MultiCrop_transform(object):
    def __init__(
            self,
            num_crops,type,normalize):

        trans=[]


        for i in range(num_crops):
            if type==0:
                weak = transforms.Compose([
                    # transforms.RandomResizedCrop(224),
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    # transforms.RandomApply([
                    #    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    # ], p=0.8),
                    # transforms.RandomGrayscale(p=0.2),
                    # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    RandAugmentMC(n=2, m=10, cutout=1),
                    #RandAugment(n=2,m=9),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])
            elif type==1:
                #not a good choice,similar result around 74.3%
                weak = transforms.Compose([
                    # transforms.RandomResizedCrop(224),
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                     transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                     ], p=0.8),
                     transforms.RandomGrayscale(p=0.2),
                     transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    #RandAugmentMC(n=2, m=10, cutout=1),
                    RandAugment(n=2, m=9),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])
            elif type==2:#very bad performance around 74.2%
                weak = transforms.Compose([
                    # transforms.RandomResizedCrop(224),
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    RandAugmentMC(n=5, m=10, cutout=1),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])
            elif type==3:
                weak = transforms.Compose([
                    # transforms.RandomResizedCrop(224),
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    RandAugmentMC(n=0, m=10, cutout=1),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])
            elif type==4:
                weak = transforms.Compose([
                    # transforms.RandomResizedCrop(224),
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    # transforms.RandomApply([
                    #    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    # ], p=0.8),
                    # transforms.RandomGrayscale(p=0.2),
                    # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    #RandAugmentMC(n=2, m=10, cutout=1),
                     RandAugment(n=2,m=9),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])
            trans.append(weak)
        self.trans=trans
        print("In total we have %d transformations"%len(self.trans))
    def __call__(self, x):
        multi_crops = list(map(lambda trans: trans(x), self.trans))
        return multi_crops

class TenCrop_transform(object):
    def __init__(
            self,normalize):

        self.trans1=transforms.Compose([
                transforms.Resize(256),
                transforms.TenCrop(224),
            ])
        self.trans2=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    def __call__(self, x):
        image=self.trans1(x)
        multi_crops=[]
        for tmp_img in image:
            tmp_img=self.trans2(tmp_img)
            multi_crops.append(tmp_img)
        return multi_crops

class Final_transform(object):
    def __init__(
            self,
            num_crops,normalize):

        trans=[]


        for i in range(num_crops):



            if i==2:
                weak = transforms.Compose([
                    # transforms.RandomResizedCrop(224),
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                     transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                     ], p=0.8),
                     transforms.RandomGrayscale(p=0.2),
                     transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    #RandAugmentMC(n=2, m=10, cutout=1),
                    #RandAugment(n=2, m=9),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])


            elif i==1:
                weak = transforms.Compose([
                    # transforms.RandomResizedCrop(224),
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    # transforms.RandomApply([
                    #    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    # ], p=0.8),
                    # transforms.RandomGrayscale(p=0.2),
                    # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    #RandAugmentMC(n=2, m=10, cutout=1),
                     RandAugment(n=2,m=9),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])
            elif i==0:
                weak = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    #transforms.RandomResizedCrop(224),
                    #transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    # transforms.RandomApply([
                    #    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    # ], p=0.8),
                    # transforms.RandomGrayscale(p=0.2),
                    # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    # RandAugmentMC(n=2, m=10, cutout=1),
                    #RandAugment(n=2, m=9),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])
            trans.append(weak)
        self.trans=trans
        print("In total we have %d transformations"%len(self.trans))
    def __call__(self, x):
        multi_crops = list(map(lambda trans: trans(x), self.trans))
        return multi_crops


class Last_transform(object):
    def __init__(
            self,
            num_crops,transform_train):

        trans=[]
        for i in range(num_crops):
            trans.append(transform_train)
        self.trans=trans
        print("In total we have %d transformations"%len(self.trans))
    def __call__(self, x):
        multi_crops = list(map(lambda trans: trans(x), self.trans))
        return multi_crops
