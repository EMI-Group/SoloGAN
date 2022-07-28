import torch
import os
import random
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def CreateDataLoader(opt):
    sourceD = [i for i in range(opt.d_num)]

    dataset = UpPairedDataset(opt.dataroot,
                            opt.crop_size,
                            opt.img_size,
                            opt.isTrain,
                            sourceD=sourceD,
                            format=opt.format)
                            
    data_loader = DataLoader(dataset=dataset,
                             batch_size=opt.batchSize,
                             shuffle=opt.isTrain,
                             drop_last=True,
                             num_workers=opt.nThreads)
                             
    return data_loader


class UpPairedDataset(Dataset):
    def __init__(self, image_path, crop_size, img_size, isTrain, sourceD=[0, 1], format='jpg'):
        self.image_path = image_path
        self.isTrain = isTrain
        self.fineSize = img_size
        self.sourceD = sourceD
        self.format = format
        print('Start preprocessing dataset..!')
        random.seed(1234)
        self.preprocess()
        print('Finished preprocessing dataset..!')
        if isTrain:     # train, add flip
            if format == 'png':
                trs = [transforms.RandomCrop(crop_size)]
                print("         $$$$$ Do not resize the image, crop_size is {} !!!!".format(crop_size)) 
                
            elif int(crop_size) == 256 and format != 'png':    # not segmentation dataset, which means, resize, then crop
                trs = [transforms.Resize(img_size, interpolation=Image.ANTIALIAS), transforms.RandomCrop(crop_size)]
                print("      $$$$$ The dataset is not hair and segmentation dataset #####")
                
            else:
                print("     ^^^^^^^ Operating on Hairs  dataset @@@@@@@")
                trs = [transforms.RandomCrop(crop_size), transforms.Resize(img_size, interpolation=Image.ANTIALIAS)]
                
            trs.append(transforms.RandomHorizontalFlip())
        else:
            if format == 'png':
                trs = [transforms.CenterCrop(crop_size)]
                
            elif int(crop_size) == 256 and format != 'png':
                trs = [transforms.Resize(img_size, interpolation=Image.ANTIALIAS), transforms.CenterCrop(crop_size)]
                
            else:
                print("     ^^^^^^^ Operating on Hairs  dataset @@@@@@@")
                trs = [transforms.CenterCrop(crop_size), transforms.Resize(img_size, interpolation=Image.ANTIALIAS)]
                
        trs.append(transforms.ToTensor())
        trs.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transforms = transforms.Compose(trs)

    def preprocess(self):
        dirs = os.listdir(self.image_path)
        trainDirs = [dir for dir in dirs if 'train' in dir]
        testDirs = [dir for dir in dirs if 'test' in dir]
        # assert len(trainDirs) < max(self.sourceD) + 2
        trainDirs.sort()  # A, B, C ..
        testDirs.sort()  # A, B, C ..
        self.filenames = []
        self.num = []
        if self.isTrain:
            for i in range(max(self.sourceD)+1):
                dir = trainDirs[i]
                filenames = glob("{}/{}/*.{}".format(self.image_path, dir, self.format))
                random.shuffle(filenames)
                self.filenames.append(filenames)
                self.num.append(len(filenames))
        else:
            for i in range(max(self.sourceD)+1):
                dir = testDirs[i]
                filenames = glob("{}/{}/*.{}".format(self.image_path, dir, self.format))
                filenames.sort()  #
                self.filenames.append(filenames)
                self.num.append(len(filenames))
        print(self.num)
        self.num_data = max(self.num)
 
    def __getitem__(self, index):
        imgs = []
        img_names = []
        for d in self.sourceD:
            index_d = index if index < self.num[d] else random.randint(0, self.num[d] - 1)
            img = Image.open(self.filenames[d][index_d]).convert('RGB')
            img = self.transforms(img)
            imgs.append(img)
            img_names.append(self.filenames[d][index_d])
        return imgs, self.sourceD, img_names

    def __len__(self):
        return self.num_data
