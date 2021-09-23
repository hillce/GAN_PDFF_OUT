import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fileDir = "./outputMAE/"
dataDir = "/home/chill/20254_2_0_NPY/"
badEids = np.load("BadEids.npy")
goodEids = np.load("GoodEids.npy")

maeList = [x for x in os.listdir(fileDir) if x[:7] in badEids]

plt.rcParams["figure.figsize"] = [6.4*4,4.8*4]
for mae in maeList:
    arr = np.load(os.path.join(fileDir,mae))

    imgs = np.load(os.path.join(dataDir,mae))

    fig,ax = plt.subplots(1,arr.shape[0])
    for i in range(arr.shape[0]):
        im = ax[i].imshow(arr[i,:,:],cmap="plasma",vmax=5)
        ax[i].axis("off")
        fig.colorbar(im,ax=ax[i],shrink=0.17)

    plt.savefig("./{}_bad.png".format(mae[:-4]))
    plt.close("all")

    fig,ax = plt.subplots(1,arr.shape[0])
    for i in range(arr.shape[0]):
        im = ax[i].imshow(imgs[i,:,:],cmap="gray")
        ax[i].axis("off")
        fig.colorbar(im,ax=ax[i],shrink=0.17)

    plt.savefig("./{}_bad_img.png".format(mae[:-4]))
