import os, sys, copy, json
import itertools

import numpy as np
import torch
import torch.nn as nn
import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomAffine

from datasets import IDEAL_Dataset, To_Tensor
from models import Generator, Discriminator

# Arg parser so I can test out different model parameters
parser = argparse.ArgumentParser(description="Training program for T1 map generation")
parser.add_argument("--dir",help="File directory for numpy images",type=str,default="D:/UKB_Liver/20254_2_0_NPY/",dest="fileDir")
parser.add_argument("--model_name",help="Name for saving the model",type=str,default="Debug",dest="modelName")
parser.add_argument("-bSize","--batch_size",help="Batch size for dataloader",type=int,default=4,dest="batchSize")
parser.add_argument("--step_size",help="Step size for learning rate decay",type=int,default=5,dest="stepSize")
parser.add_argument("--gpu_choice",help="Which gpu to run on, on OCMR",type=str,default=None,dest="deviceChoice")

args = parser.parse_args()

fileDir = args.fileDir
modelName = args.modelName
bSize = args.batchSize
stepSize = args.stepSize
device = args.deviceChoice

toT = To_Tensor()
trnsInTest = transforms.Compose([toT])

testEids = np.load("testSet.npy")
datasetTest = IDEAL_Dataset("testSet.npy",fileDir=fileDir,transform=trnsInTest)
loaderTest = DataLoader(datasetTest,batch_size=bSize,shuffle=False,pin_memory=False)

device = torch.device(device)

modelDict = torch.load("./TrainingLogs/{}/model.pt".format(modelName),map_location="cpu")

gDict = modelDict["Generator_state_dict"]

# keys = list(gDict.keys())
# for k in keys:
#     gDict[k[7:]] = gDict[k]
#     del gDict[k]

netG = Generator(6,232,256,outC=6)
netG.load_state_dict(gDict)
del modelDict

netG = netG.to(device)
testLen = datasetTest.__len__()

maeLoss = nn.L1Loss(reduction="none")
mseLoss = nn.MSELoss(reduction="none")

knNums = list(itertools.combinations_with_replacement([0,1,2,3,4,5],2))
with torch.no_grad():
    for ii,data in enumerate(loaderTest):

        sys.stdout.write("\r[{}/{}]".format(ii*bSize,testLen))

        eids = testEids[(ii*bSize):(ii*bSize+bSize)]
        maeErr = np.zeros((bSize,6,6,6)) # ch,kn0,kn1
        mseErr = np.zeros((bSize,6,6,6))
        for kn0,kn1 in knNums:
            knImgs = copy.deepcopy(data).numpy()
            for j in range(knImgs.shape[0]):
                knImgs[j,(kn0,kn1),:,:] = 0.0

            knImgs = torch.from_numpy(knImgs)

            inpData = data.to(device)
            knImgs = knImgs.to(device)

            fake = netG(knImgs)

            mae = maeLoss(fake,inpData)
            mse = mseLoss(fake,inpData)

            mae = mae.cpu().numpy()
            mse = mse.cpu().numpy()

            maeErr[:,:,kn0,kn1] = np.mean(mae,axis=(2,3))
            mseErr[:,:,kn0,kn1] = np.mean(mse,axis=(2,3))

        for idx,eid in enumerate(eids):
            np.save("./outputMAE/{}".format(eid),maeErr[idx,:,:,:])
            np.save("./outputMSE/{}".format(eid),mseErr[idx,:,:,:])

