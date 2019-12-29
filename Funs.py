import numpy as np
import scipy.stats as st

# Generate a Gaussian kernel 
# rbf = gkern(5,2.5)*273
def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

# plot confusion matrix
def plot_confusion_matrix(cfm,name,dpi=80):
    lcz = ['LCZ-1','LCZ-2','LCZ-3','LCZ-4','LCZ-5','LCZ-6','LCZ-7','LCZ-8','LCZ-9',
           'LCZ-10','LCZ-A','LCZ-B','LCZ-C','LCZ-D','LCZ-E','LCZ-F','LCZ-G']
    plt.figure(figsize=(12,10),dpi=dpi)
    imx = cfm.shape[0]
    cm = np.zeros([imx,imx])
    for i in range(imx):
        cm[:,i] = cfm[:,i]/cfm[:,i].sum()*100
    plt.imshow(cm,interpolation='nearest')
    plt.title(name)
    plt.colorbar()
    
    tick_marks=np.arange(imx)
    plt.xticks(tick_marks,np.arange(imx)+1,fontsize=6,rotation=45)
    plt.yticks(tick_marks,np.arange(imx)+1,fontsize=6,rotation=45)
    plt.ylabel('Predicted Label')
    plt.xlabel('Reference Label')
    fmt = '.0f'
    thresh = cm.max() / 2.
    for i in range(imx):
        for j in range(imx):
            plt.text(j, i, format(cfm[i,j], fmt),
                    ha="center", va="center",fontsize=10,
                    color="white" if cm[i,j] < thresh else "black")
    plt.xlim(-0.5,imx-0.5)
    plt.ylim(imx-0.5,-0.5)
    plt.xticks(np.arange(0,17),lcz)
    plt.yticks(np.arange(0,17),lcz)
    plt.tight_layout()
    plt.savefig(name+'.pdf')
    plt.show()
    
def setGeo(geotransform,bgx,bgy):
    reset0 = geotransform[0] + bgx*geotransform[1]
    reset3 = geotransform[3] + bgy*geotransform[5]
    reset = (reset0,geotransform[1],geotransform[2],
             reset3,geotransform[4],geotransform[5])
    return reset

