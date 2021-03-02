from __future__ import print_function
import torch
import torch.utils.data as data
from PIL import Image, PILLOW_VERSION
import os
import csv



class Places205(data.Dataset):
    def __init__(self, root, split, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.data_folder  = os.path.join(self.root, 'data', 'vision', 'torralba', 'deeplearning', 'images256')
        self.split_folder = os.path.join(self.root, 'trainvalsplit_places205')
        assert(split=='train' or split=='val')
        split_csv_file = os.path.join(self.split_folder, split+'_places205.csv')

        self.transform = transform
        self.target_transform = target_transform
        self.img_files = []
        self.labels = []
        with open(split_csv_file, 'r') as file:
            line=file.readline()
            while line:
                line=line.strip("\n")
                split_result=line.split()
                self.img_files.append(split_result[0])
                self.labels.append(int(split_result[1]))
                line=file.readline()
        # with open(split_csv_file, 'rb') as f:
        #     reader = csv.reader(f, delimiter=' ')
        #     self.img_files = []
        #     self.labels = []
        #     for row in reader:
        #         self.img_files.append(row[0])
        #         self.labels.append(long(row[1]))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image_path = os.path.join(self.data_folder, self.img_files[index])
        img = Image.open(image_path).convert('RGB')
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.labels)