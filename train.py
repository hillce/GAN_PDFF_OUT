import os, sys, copy, json

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
parser.add_argument("--model_name",help="Name for saving the model",type=str,dest="modelName",required=True)
parser.add_argument("-lr",help="Learning rate for the optimizer",type=float,default=1e-3,dest="lr")
parser.add_argument("-b1",help="Beta 1 for the Adam optimizer",type=float,default=0.9,dest="b1")
parser.add_argument("-bSize","--batch_size",help="Batch size for dataloader",type=int,default=4,dest="batchSize")
parser.add_argument("-nE","--num_epochs",help="Number of Epochs to train for",type=int,default=50,dest="numEpochs")
parser.add_argument("--step_size",help="Step size for learning rate decay",type=int,default=5,dest="stepSize")
parser.add_argument("--gpu_choice",help="Which gpu to run on, on OCMR",type=str,default=None,dest="deviceChoice")

args = parser.parse_args()

fileDir = args.fileDir
modelName = args.modelName
lr = args.lr
b1 = args.b1
bSize = args.batchSize
numEpochs = args.numEpochs
stepSize = args.stepSize
device = args.deviceChoice

modelDir = "./TrainingLogs/{}/".format(modelName)
os.makedirs(modelDir)

hParamDict = {}
hParamDict["fileDir"] = fileDir
hParamDict["modelName"] = modelName
hParamDict["lr"] = lr
hParamDict["b1"] = b1
hParamDict["batchSize"] = bSize
hParamDict["numEpochs"] = numEpochs
hParamDict["stepSize"] = stepSize
hParamDict["deviceChoice"] = device

with open("{}hparams.json".format(modelDir),"w") as f:
    json.dump(hParamDict,f)

writer = SummaryWriter("{}tensorboard".format(modelDir))

rA = RandomAffine(5,translate=(0.01,0.01),shear=5)
toT = To_Tensor()

trnsInTrain = transforms.Compose([toT,rA])
trnsInVal = transforms.Compose([toT])

datasetTrain = IDEAL_Dataset("trainSet.npy",fileDir=fileDir,transform=trnsInTrain)
datasetVal = IDEAL_Dataset("valSet.npy",fileDir=fileDir,transform=trnsInVal)

loaderTrain = DataLoader(datasetTrain,batch_size=bSize,shuffle=True,pin_memory=False)
loaderVal = DataLoader(datasetVal,batch_size=bSize,shuffle=False,pin_memory=False)

device = torch.device(device)

netG = Generator(6,232,256,outC=6)
netD = Discriminator(6,232,256)

if torch.cuda.device_count() > 1:
    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)

netG = netG.to(device)
netD = netD.to(device)

real_label = 1
fake_label = 0

gen_loss = nn.SmoothL1Loss()
disc_loss = nn.BCELoss()

optim_G = torch.optim.Adam(netG.parameters(), lr=lr, betas=(b1,0.999))
optim_D = torch.optim.Adam(netD.parameters(), lr=lr, betas=(b1,0.999))

trainLen = datasetTrain.__len__()
valLen = datasetVal.__len__()

lowestLoss = 1000000000000000.0
trainLossCnt = 0
valLossCnt = 0

for epoch in range(numEpochs):
    print("\n","#"*50)
    print("Epoch {}".format(epoch))
    print("#"*50,"\n")

    netG.train()
    netD.train()
    runningLoss = 0.0
    print("Training: \n")
    for ii, data in enumerate(loaderTrain):

        optim_G.zero_grad()
        optim_D.zero_grad()

        knImgs = copy.deepcopy(data).numpy()
        for j in range(knImgs.shape[0]):
            knNum = np.random.randint(knImgs.shape[1],size=np.random.randint(2)+1)
            knImgs[j,knNum,:,:] = 0.0
        knImgs = torch.from_numpy(knImgs)

        inpData = data.to(device)
        knImgs = knImgs.to(device)

        netD.zero_grad()

        label = torch.full((inpData.size()[0],),real_label,device=device,dtype=torch.float)
        real_out = netD(inpData).view(-1)
        errD_real = disc_loss(real_out,label)
        errD_real.backward()

        netG.zero_grad()

        fake = netG(knImgs)

        label.fill_(fake_label)
        fake_out = netD(fake.detach()).view(-1)
        errD_fake = disc_loss(fake_out,label)
        errD_fake.backward()

        optim_D.step()

        errG_recon = gen_loss(fake,inpData)
        errG_recon.backward(retain_graph=True)

        label.fill_(real_label)
        fake_out_G = netD(fake).view(-1)
        errG_fake = disc_loss(fake_out_G, label)
        errG_fake.backward()

        optim_G.step()

        loss_batch = errG_recon.item()
        runningLoss += loss_batch

        sys.stdout.write("\r[{}/{}] Loss: {:.5f}".format(ii*bSize,trainLen,runningLoss/(ii+1)))

        writer.add_scalar('Loss/train',loss_batch,trainLossCnt)

        trainLossCnt += 1

        # torch.cuda.empty_cache()

    with torch.no_grad():
        valLoss = 0.0
        netG.eval()
        netD.eval()

        print("\nValidating: \n")
        for ii, data in enumerate(loaderVal):

            knImgs = copy.deepcopy(data).numpy()
            for j in range(knImgs.shape[0]):
                knNum = np.random.randint(knImgs.shape[1],size=np.random.randint(3)+1)
                knImgs[j,knNum,:,:] = 0.0
            knImgs = torch.from_numpy(knImgs)

            inpData = data.to(device)
            knImgs = knImgs.to(device)

            fake = netG(knImgs)
            errG_recon = gen_loss(fake,inpData)
            loss_batch = errG_recon.item()

            valLoss += loss_batch

            writer.add_scalar("Loss/val",loss_batch,valLossCnt)

            sys.stdout.write("\r[{}/{}] Loss: {:.5f}".format(ii*bSize,valLen,valLoss/(ii+1)))

            valLossCnt += 1

        valLoss /= valLen

        if epoch == 0:
            torch.save({"Epoch":epoch+1,
            "Generator_state_dict":netG.module.state_dict(),
            "Discriminator_state_dict":netD.module.state_dict(),
            "Generator_optimizer":optim_G.state_dict(),
            "Discriminator_optimizer":optim_D.state_dict(),
            "Val_loss":valLoss,
            "Train_cnt":trainLossCnt,
            "Val_cnt":valLossCnt,
            },"{}model.pt".format(modelDir))
            lowestLoss = valLoss
        else:
            if valLoss < lowestLoss:
                sys.stdout.write("\nvalLoss {} < {} lowestLoss. Saving!\n".format(valLoss,lowestLoss))

                torch.save({"Epoch":epoch+1,
                "Generator_state_dict":netG.module.state_dict(),
                "Discriminator_state_dict":netD.module.state_dict(),
                "Generator_optimizer":optim_G.state_dict(),
                "Discriminator_optimizer":optim_D.state_dict(),
                "Val_loss":valLoss,
                "Train_cnt":trainLossCnt,
                "Val_cnt":valLossCnt,
                },"{}model.pt".format(modelDir))
                lowestLoss = valLoss  