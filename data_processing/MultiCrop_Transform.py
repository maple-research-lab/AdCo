#modified from https://github.com/facebookresearch/swav/blob/master/src/multicropdataset.py
from torchvision import transforms
from data_processing.loader import GaussianBlur

class Multi_Transform(object):
    def __init__(
            self,
            size_crops,
            nmb_crops,
            min_scale_crops,
            max_scale_crops,normalize,init_size=224):
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        trans=[]
        #image_k
        weak = transforms.Compose([
            transforms.RandomResizedCrop(init_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        trans.append(weak)


        trans_weak=[]

        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )


            weak=transforms.Compose([
            randomresizedcrop,
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])
            trans_weak.extend([weak]*nmb_crops[i])

        trans.extend(trans_weak)
        self.trans=trans
        print("in total we have %d transforms"%(len(self.trans)))
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
