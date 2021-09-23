import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fileDir = "./outputMAE/"
maeList = [os.path.join(fileDir,x) for x in os.listdir(fileDir)]

for mae in maeList:
    arr = np.load(mae)

    fig,ax = plt.subplots(1,arr.shape[0])
    for i in range(arr.shape[0]):
        im = ax[i].imshow(arr[i,:,:],cmap="plasma",vmax=3)
        ax[i].axis("off")
        fig.colorbar(im,ax=ax[i],shrink=0.17)

    plt.savefig("./tempFig.png")
    
