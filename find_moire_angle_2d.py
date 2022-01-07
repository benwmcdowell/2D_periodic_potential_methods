import matplotlib.pyplot as plt
import numpy as np

#elv is the experimental lattice vectors to be matched
#lv1 and lv2 are the theoretical lattice vectors defining the moire pattern, which form an incident angle of theta
def plot_moire_angle(elv,lv1,name1,lv2,name2,**args):
    npts=361
    theta=np.array([i/(npts-1)*360 for i in range(npts)])
    theta*=np.pi/180
    
    tempvar=np.zeros((3,3))
    tempvar[:2,:2]+=elv
    tempvar[2,2]=1.0
    elv[0]=(2*np.pi*np.cross(tempvar[1],tempvar[2])/np.dot(tempvar[0],np.cross(tempvar[1],tempvar[2])))[:2]
    elv[1]=(2*np.pi*np.cross(tempvar[2],tempvar[0])/np.dot(tempvar[0],np.cross(tempvar[1],tempvar[2])))[:2]
    
    tempvar=np.zeros((3,3))
    tempvar[:2,:2]+=lv1
    tempvar[2,2]=1.0
    lv1[0]=(2*np.pi*np.cross(tempvar[1],tempvar[2])/np.dot(tempvar[0],np.cross(tempvar[1],tempvar[2])))[:2]
    lv1[1]=(2*np.pi*np.cross(tempvar[2],tempvar[0])/np.dot(tempvar[0],np.cross(tempvar[1],tempvar[2])))[:2]
    
    tempvar=np.zeros((3,3))
    tempvar[:2,:2]+=lv2
    tempvar[2,2]=1.0
    lv2[0]=(2*np.pi*np.cross(tempvar[1],tempvar[2])/np.dot(tempvar[0],np.cross(tempvar[1],tempvar[2])))[:2]
    lv2[1]=(2*np.pi*np.cross(tempvar[2],tempvar[0])/np.dot(tempvar[0],np.cross(tempvar[1],tempvar[2])))[:2]
    
    #heirarchy of diff is (experimental lattice vector, theoretical lattice vector, +/-)
    diff=np.zeros((2,2,2,2,npts))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(npts):
                        rot=np.array([[np.cos(theta[m]),-np.sin(theta[m])],[np.sin(theta[m]),np.cos(theta[m])]])
                        if l==0:
                            sf=1
                        if l==1:
                            sf=-1
                        diff[i,j,k,l,m]+=np.linalg.norm(lv1[j]+sf*np.dot(lv2[k],rot)-elv[i])
                        
    theta*=180/np.pi
    
    fig,axs=plt.subplots(4,2,sharex=True)
    for j in range(2):
        for x in range(4):
            if x==0:
                k=0
                l=0
            if x==1:
                k=0
                l=1
            if x==2:
                k=1
                l=0
            if x==3:
                k=1
                l=1
            
            axs[x,j].plot(theta,diff[0,j,k,l],label='$V_{e1}$')
            axs[x,j].plot(theta,diff[1,j,k,l],label='$V_{e2}$')
        axs[-1,j].set(xlabel='theta / degrees')
        
    axs[0,0].set(title='V1 {}'.format(name1))
    axs[0,1].set(title='V2 {}'.format(name1))
    axs[0,0].set(ylabel='+V1 {}'.format(name2))
    axs[1,0].set(ylabel='-V1 {}'.format(name2))
    axs[2,0].set(ylabel='+V2 {}'.format(name2))
    axs[3,0].set(ylabel='-V2 {}'.format(name2))
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.show()