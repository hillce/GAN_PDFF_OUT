import os, copy, json, sys, re, warnings
from numpy.lib.type_check import imag

import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

class IDEAL_Dataset(Dataset):
    """
    T1 Dataset
    """
    def __init__(self,loadDir,fileDir="D:/UKB_Liver/20254_2_0_NPY/",transform=None):
        self.fileDir = fileDir
        self.transform = transform
        self.dataSet = np.load(loadDir)

    def __getitem__(self, index):
        inpData = np.load("{}{}".format(self.fileDir,self.dataSet[index]))
        sample = inpData

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.dataSet)

class To_Tensor(object):

    def __call__(self,inpData):
        return torch.from_numpy(inpData).float() 